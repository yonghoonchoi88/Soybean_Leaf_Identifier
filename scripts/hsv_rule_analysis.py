"""[개선 분석 ②+④] OpenCV HSV 병반 후보 분석 + 규칙 기반 보조 판단 오프라인 시뮬레이션 (별도 분석 도구).

- 제공 코드·서비스 API는 수정하지 않는다. 이미 저장된 예측 결과(evaluate_split.py CSV)와
  이미지에서 계산한 HSV 특징만으로 "규칙을 붙였다면 어땠을까"를 시뮬레이션한다.
- 규칙 기준값은 Validation 으로만 정하고, Test 에는 그 기준을 그대로 한 번 적용해 보고만 한다.

처리 단계 (교과 9 OpenCV)
  1) 256×256 축소 → BGR→HSV (cv2.cvtColor)
  2) 잎 후보: 녹색 범위 cv2.inRange → Opening(작은 잡음 제거) → 잎 안쪽 구멍 채우기(병반 포함)
     ※ 사진 테두리와 이어진 흙·배경은 구멍으로 보지 않아 제외된다
  3) 병반 후보: 잎 영역 안에서 녹색이 아니고 갈색·황갈색(H 5~24) 인 픽셀
  4) lesion_ratio = 병반 후보 픽셀 / 잎 픽셀,  lesion_hue = 병반 후보의 H 중앙값

규칙 (보조 판단: 라벨을 바꾸지 않고 "확인 필요" 표시만 붙임)
  R1 healthy 의심 : 예측이 healthy 인데 lesion_ratio > T1
                    T1 = Validation 에서 "healthy를 맞힌 사진"의 lesion_ratio 95백분위 (정상 잎의 5%만 표시되도록)
  R2 저신뢰       : 1·2위 점수 차(margin) < M  (M = 0.4, E1 Validation margin 분석에서 0.4 이상 구간 정확도 97.6%)
  R1 ∪ R2        : 둘 중 하나라도 해당하면 확인 필요

사용 예:
  python scripts/hsv_rule_analysis.py
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIZE = 256
GREEN_LO, GREEN_HI = (25, 25, 35), (95, 255, 255)     # 잎(녹색) HSV 범위 (OpenCV H: 0~179)
BROWN_LO, BROWN_HI = (5, 60, 40), (24, 255, 235)      # 병반 후보(갈색·황갈색) HSV 범위
LABEL_H = 34


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """녹색 마스크 안쪽의 구멍(병반 자리)을 채운다. 사진 테두리와 이어진 배경은 채우지 않는다."""
    h, w = mask.shape
    holes = cv2.bitwise_not(mask)
    flood = np.zeros((h + 2, w + 2), np.uint8)
    border = [(x, y) for x in range(w) for y in (0, h - 1)] + [(x, y) for y in range(h) for x in (0, w - 1)]
    for x, y in border:
        if holes[y, x] == 255:
            cv2.floodFill(holes, flood, (x, y), 0)        # 테두리와 연결된 비녹색 = 배경 → 지움
    return cv2.bitwise_or(mask, holes)                     # 남은 것 = 잎 안쪽 구멍


def hsv_features(path: Path):
    bgr = cv2.resize(cv2.imread(str(path)), (SIZE, SIZE))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    leaf = fill_holes(green)
    brown = cv2.inRange(hsv, BROWN_LO, BROWN_HI)
    lesion = cv2.bitwise_and(brown, cv2.bitwise_and(leaf, cv2.bitwise_not(green)))
    leaf_px, lesion_px = int((leaf > 0).sum()), int((lesion > 0).sum())
    hue = float(np.median(hsv[..., 0][lesion > 0])) if lesion_px else None
    feats = {"leaf_area": round(leaf_px / SIZE ** 2, 4),
             "lesion_ratio": round(lesion_px / leaf_px, 4) if leaf_px else 0.0,
             "lesion_hue": hue}
    return feats, bgr, leaf, lesion


def read_predictions(csv_path: Path):
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["correct"] = r["correct"] == "True"
        r["top1_score"] = float(r["top1_score"])
        r["margin"] = float(r["margin"])
    return rows


def add_features(rows, data_dir: Path, split: str):
    for r in rows:
        feats, *_ = hsv_features(data_dir / split / r["true"] / r["file"])
        r.update(feats)
    return rows


def q(values, p):
    values = sorted(values)
    if not values:
        return None
    k = (len(values) - 1) * p
    lo, hi = int(np.floor(k)), int(np.ceil(k))
    return round(values[lo] + (values[hi] - values[lo]) * (k - lo), 4)


def class_stats(rows):
    stats = {}
    for c in sorted({r["true"] for r in rows}):
        g = [r for r in rows if r["true"] == c]
        ratios = [r["lesion_ratio"] for r in g]
        hues = [r["lesion_hue"] for r in g if r["lesion_hue"] is not None]
        stats[c] = {"n": len(g), "lesion_ratio_p25": q(ratios, 0.25), "lesion_ratio_median": q(ratios, 0.5),
                    "lesion_ratio_p75": q(ratios, 0.75),
                    "lesion_hue_median": round(statistics.median(hues), 1) if hues else None}
    return stats


def apply_rules(rows, t1, m):
    for r in rows:
        r["R1"] = r["pred"] == "healthy" and r["lesion_ratio"] > t1
        r["R2"] = r["margin"] < m
        r["review"] = r["R1"] or r["R2"]
    out = {}
    for name in ("R1", "R2", "review"):
        flagged = [r for r in rows if r[name]]
        auto = [r for r in rows if not r[name]]
        wrong = [r for r in rows if not r["correct"]]
        out[name] = {"flagged": len(flagged), "flag_rate": round(len(flagged) / len(rows), 4),
                     "errors_total": len(wrong), "errors_caught": sum(1 for r in wrong if r[name]),
                     "auto_accuracy": round(sum(r["correct"] for r in auto) / len(auto), 4) if auto else None,
                     "auto_count": len(auto)}
    return out


def font(size=13):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if Path(path).exists():
        return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def panel(path: Path, r: dict):
    feats, bgr, leaf, lesion = hsv_features(path)
    over = bgr.copy()
    over[lesion > 0] = (0, 0, 255)                         # 병반 후보 = 빨강
    imgs = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cv2.cvtColor(leaf, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(over, cv2.COLOR_BGR2RGB)]
    texts = [[path.name[:32], f"true {r['true']}"],
             ["leaf mask", f"leaf area {feats['leaf_area']:.2f}"],
             [f"pred {r['pred']} {r['top1_score']:.2f}", f"lesion {feats['lesion_ratio']:.4f} hue {feats['lesion_hue']}"]]
    canvas = Image.new("RGB", (SIZE * 3 + 12, SIZE + LABEL_H), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (arr, lines) in enumerate(zip(imgs, texts)):
        canvas.paste(Image.fromarray(arr), (i * (SIZE + 6), 0))
        color = (30, 30, 30) if i < 2 else ((0, 110, 0) if r["correct"] else (190, 0, 0))
        for j, t in enumerate(lines):
            draw.text((i * (SIZE + 6) + 4, SIZE + 2 + j * 16), t, fill=color, font=font())
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-val", default="reports/data/eval_val_E3_full_aug_predictions.csv")
    parser.add_argument("--pred-test", default="reports/data/eval_test_E3_full_aug_predictions.csv")
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--margin", type=float, default=0.4)
    parser.add_argument("--out-data", default="reports/data")
    parser.add_argument("--out-fig", default="reports/figures/hsv")
    args = parser.parse_args()

    data = Path(args.data)
    val = add_features(read_predictions(Path(args.pred_val)), data, "val")
    test = add_features(read_predictions(Path(args.pred_test)), data, "test")

    # ---- ② 클래스별 병반 특징 (Validation) ----
    stats_val = class_stats(val)
    # ---- ④ 규칙 기준값은 Validation 으로만 정한다 ----
    healthy_ok = [r["lesion_ratio"] for r in val if r["true"] == "healthy" and r["pred"] == "healthy"]
    t1 = q(healthy_ok, 0.95)
    if t1 is None:   # (예외) healthy 정답 사례가 없으면 R1 은 꺼 둔다
        t1 = float("inf")
    rules_val = apply_rules(val, t1, args.margin)
    rules_test = apply_rules(test, t1, args.margin)   # 같은 기준을 Test 에 1회 적용 (기준 수정 없음)

    out_data = Path(args.out_data)
    for name, rows in (("val", val), ("test", test)):
        with open(out_data / f"hsv_features_{name}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summary = {"model_predictions": {"val": args.pred_val, "test": args.pred_test},
               "hsv_ranges": {"green": [GREEN_LO, GREEN_HI], "brown": [BROWN_LO, BROWN_HI]},
               "class_stats_val": stats_val, "class_stats_test": class_stats(test),
               "thresholds_from_val": {"T1_lesion_ratio": t1, "M_margin": args.margin},
               "rules_val(design)": rules_val, "rules_test(apply once)": rules_test}
    (out_data / "hsv_rule_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 그림: 오답 전부 + 클래스별 정답 예시 1장 ----
    out_fig = Path(args.out_fig)
    out_fig.mkdir(parents=True, exist_ok=True)
    picks = [("val", r) for r in val if not r["correct"]] + [("test", r) for r in test if not r["correct"]]
    for c in ("angular_leaf_spot", "bean_rust", "healthy"):
        ok = [r for r in test if r["true"] == c and r["correct"]]
        if ok:
            picks.append(("test", ok[0]))
    panels = []
    for split, r in picks:
        p = panel(data / split / r["true"] / r["file"], r)
        p.save(out_fig / f"hsv_{split}_{Path(r['file']).stem}.png")
        panels.append(p)
    sheet = Image.new("RGB", (panels[0].width, sum(p.height + 8 for p in panels)), "white")
    y = 0
    for p in panels:
        sheet.paste(p, (0, y))
        y += p.height + 8
    sheet.save(out_fig / "hsv_summary.png")

    # ---- 터미널 요약 (증빙 캡처용) ----
    print("② 클래스별 병반 후보 특징 (Validation, 정답 라벨 기준)")
    print(f"  {'class':<20}{'n':>4}{'ratio p25':>11}{'median':>9}{'p75':>9}{'hue med':>9}")
    for c, s in stats_val.items():
        print(f"  {c:<20}{s['n']:>4}{s['lesion_ratio_p25']:>11}{s['lesion_ratio_median']:>9}"
              f"{s['lesion_ratio_p75']:>9}{str(s['lesion_hue_median']):>9}")
    print(f"\n④ 규칙 기준 (Validation 으로만 결정): T1 lesion_ratio > {t1}  /  M margin < {args.margin}")
    print(f"  {'split':<6}{'rule':<8}{'보류':>6}{'보류율':>8}{'오답':>6}{'잡은 오답':>9}{'자동판정 정확도':>14}")
    for split, res in (("val", rules_val), ("test", rules_test)):
        for name, v in res.items():
            print(f"  {split:<6}{name:<8}{v['flagged']:>6}{v['flag_rate']:>8.1%}{v['errors_total']:>6}"
                  f"{v['errors_caught']:>9}{str(v['auto_accuracy']):>14}")
    print("\n오답 사진의 HSV 특징")
    for split, r in picks:
        if not r["correct"]:
            print(f"  {split:<5}{r['file']:<30} true={r['true']:<18} pred={r['pred']:<18}"
                  f" ratio={r['lesion_ratio']:<7} hue={r['lesion_hue']}  R1={r['R1']} R2={r['R2']}")
    print(f"\n저장: {out_data}/hsv_features_val.csv, hsv_features_test.csv, hsv_rule_summary.json, {out_fig}/hsv_summary.png")


if __name__ == "__main__":
    main()
