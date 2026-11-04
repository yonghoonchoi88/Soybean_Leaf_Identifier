"""학습된 모델을 dataset/<split> 전체 이미지로 평가하는 별도 도구 (제공 코드는 수정하지 않음).

- 서비스(ai/hf_model.py)와 같은 방식: transformers pipeline("image-classification")
- 정답 라벨 = 이미지가 들어 있는 폴더 이름 (평가 가이드 규칙)
- 출력: 정확도, 혼동행렬, 클래스별 Precision/Recall/F1, 1·2위 점수 차(margin) 분석, 오분류 목록

사용 예:
  # 1) Validation: 규칙(임계값) 설계·모델 선택은 여기서만 한다
  python scripts/evaluate_split.py --model artifacts/beans_resnet18 --split val --tag headonly
  # 2) Test: 모델을 확정한 뒤 "한 번만" 확인한다 (Test 결과로 모델/임계값을 고르지 않는다)
  python scripts/evaluate_split.py --model artifacts/beans_resnet18 --split test --tag headonly
"""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".webp"}  # finetune.py / check_dataset.py와 같은 확장자 기준


def collect_images(split_dir: Path):
    """split 폴더 아래 '클래스 폴더/이미지' 목록을 (경로, 정답라벨)로 모은다."""
    classes = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    items = []
    for cls in classes:
        for f in sorted((split_dir / cls).iterdir()):
            if f.is_file() and f.suffix.lower() in EXTS:
                items.append((f, cls))
    return classes, items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/beans_resnet18", help="평가할 모델 폴더")
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--tag", default="headonly", help="결과 파일 이름에 붙일 이름 (예: headonly, full)")
    parser.add_argument("--device", type=int, default=-1, help="-1=CPU(서비스와 동일), 0=GPU")
    parser.add_argument("--out", default="reports/data")
    args = parser.parse_args()

    from transformers import pipeline  # 무거운 import는 실행 시점에

    split_dir = Path(args.data) / args.split
    classes, items = collect_images(split_dir)
    print(f"[평가] model={args.model} split={args.split} images={len(items)} classes={classes}")

    clf = pipeline("image-classification", model=args.model, device=args.device)

    rows = []
    started = time.perf_counter()
    for path, true_label in items:
        with Image.open(path) as raw:
            image = raw.convert("RGB")
        preds = clf(image, top_k=len(classes))
        preds = sorted(preds, key=lambda x: x["score"], reverse=True)
        top1, top2 = preds[0], preds[1]
        rows.append({
            "file": path.name,
            "true": true_label,
            "pred": top1["label"],
            "correct": top1["label"] == true_label,
            "top1_score": round(float(top1["score"]), 4),
            "top2_label": top2["label"],
            "top2_score": round(float(top2["score"]), 4),
            "margin": round(float(top1["score"]) - float(top2["score"]), 4),  # 1위-2위 점수 차
        })
    elapsed = time.perf_counter() - started

    # ---- 지표 계산 (외부 라이브러리 없이 직접 계산) ----
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    # 혼동행렬: confusion[정답][예측] = 개수
    confusion = {t: {p: 0 for p in classes} for t in classes}
    for r in rows:
        confusion[r["true"]][r["pred"]] += 1

    per_class = {}
    for c in classes:
        tp = confusion[c][c]
        fp = sum(confusion[t][c] for t in classes if t != c)   # c라고 예측했지만 틀림
        fn = sum(confusion[c][p] for p in classes if p != c)   # 정답 c를 놓침
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = {"support": tp + fn, "predicted": tp + fp,
                        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

    # ---- margin 구간별 정확도: "애매한 예측일수록 더 많이 틀리는가?" ----
    bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1.01)]
    margin_bins = []
    for lo, hi in bins:
        sel = [r for r in rows if lo <= r["margin"] < hi]
        acc = sum(r["correct"] for r in sel) / len(sel) if sel else None
        margin_bins.append({"range": f"{lo:.1f}~{min(hi, 1.0):.1f}", "count": len(sel),
                            "accuracy": round(acc, 4) if acc is not None else None})

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "split": args.split,
        "tag": args.tag,
        "device": "cpu" if args.device < 0 else f"cuda:{args.device}",
        "images": n,
        "correct": correct,
        "accuracy": round(correct / n, 4) if n else None,
        "classes": classes,
        "confusion_matrix(row=true,col=pred)": confusion,
        "per_class": per_class,
        "margin_bins": margin_bins,
        "misclassified": [r for r in rows if not r["correct"]],
        "elapsed_sec": round(elapsed, 2),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"eval_{args.split}_{args.tag}"
    (out / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / f"{stem}_predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---- 터미널 요약 (증빙 캡처용) ----
    print(f"\n정확도(Accuracy): {correct}/{n} = {report['accuracy']}")
    print("\n혼동행렬 (행=정답, 열=예측)")
    w = max(len(c) for c in classes) + 2
    print(" " * w + "".join(f"{c[:12]:>14}" for c in classes))
    for t in classes:
        print(f"{t:<{w}}" + "".join(f"{confusion[t][p]:>14}" for p in classes))
    print("\n클래스별 지표")
    for c, m in per_class.items():
        print(f"  {c:<20} P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  (정답 {m['support']}장, 예측 {m['predicted']}장)")
    print("\n1·2위 점수 차(margin) 구간별 정확도")
    for b in margin_bins:
        print(f"  margin {b['range']}: {b['count']:>3}장, 정확도 {b['accuracy']}")
    print(f"\n오분류 {len(report['misclassified'])}장 / 소요 {report['elapsed_sec']}초")
    print(f"저장: {out / (stem + '.json')}, {out / (stem + '_predictions.csv')}")


if __name__ == "__main__":
    main()
