"""[개선 분석 ③] Grad-CAM: 모델이 이미지의 "어디를 보고" 판단했는지 히트맵으로 그리는 별도 분석 도구.

- 제공 코드는 수정하지 않는다. 추가 설치 없이 torch · transformers · PIL · numpy만 사용한다.
- 정확도를 높이는 기능이 아니라, 오분류 원인을 눈으로 확인하는 "설명·오류 분석 도구"다.

원리 (Grad-CAM)
  1) ResNet의 마지막 합성곱 단계(stage 4) 출력 = 512장의 7×7 특징 지도
  2) 특정 클래스 점수를 그 특징 지도로 미분 → 각 특징 지도가 그 클래스 판단에 얼마나 기여했는지(가중치)
  3) 가중치 × 특징 지도를 더하고 음수는 버림(ReLU) → 7×7 "근거 지도" → 224×224로 키워 사진 위에 겹침
  빨간 곳 = 그 클래스라고 판단하는 데 크게 기여한 영역

사용 예:
  # E3의 Test·Val 오답 전부 + 같은 사진을 E0는 어떻게 봤는지 비교
  python scripts/gradcam.py --model artifacts/beans_resnet18_full_aug \
      --from-eval reports/data/eval_test_E3_full_aug.json reports/data/eval_val_E3_full_aug.json \
      --compare-model artifacts/beans_resnet18
  # 원하는 사진을 직접 지정 (정답을 모르면 true 칸은 비움)
  python scripts/gradcam.py --model artifacts/beans_resnet18_full_aug --images dataset/test/healthy/healthy_test.0.jpg
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModelForImageClassification

SIZE = 224          # 모델 입력 크기 (전처리 후)
LABEL_H = 34        # 그림 아래 글자 영역 높이


class GradCAM:
    def __init__(self, model_dir: str):
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.model = AutoModelForImageClassification.from_pretrained(model_dir).eval()
        self.id2label = self.model.config.id2label
        self.label2id = {v: int(k) for k, v in self.id2label.items()}
        self._acts, self._grads = None, None
        # HF ResNet 구조: model.resnet.encoder.stages[-1] = 마지막 합성곱 단계 (출력 512×7×7)
        target = self.model.resnet.encoder.stages[-1]
        target.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        self._acts = output                                   # 순전파 때 특징 지도 저장
        output.register_hook(lambda g: setattr(self, "_grads", g))  # 역전파 때 기울기 저장

    def run(self, image: Image.Image):
        """반환: 모델 입력 그림(224×224 RGB), 클래스별 점수(dict), cam 함수"""
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        logits = self.model(pixel_values=pixel_values).logits      # 기울기 계산을 위해 no_grad 사용 안 함
        probs = torch.softmax(logits, dim=-1)[0].detach()
        scores = {self.id2label[i]: float(probs[i]) for i in range(len(probs))}

        def cam_for(label: str) -> np.ndarray:
            self.model.zero_grad()
            logits[0, self.label2id[label]].backward(retain_graph=True)
            weights = self._grads.mean(dim=(2, 3), keepdim=True)    # 특징 지도별 중요도
            cam = torch.relu((weights * self._acts).sum(dim=1))[0].detach().numpy()
            cam = cam / cam.max() if cam.max() > 0 else cam
            heat = Image.fromarray(np.uint8(cam * 255)).resize((SIZE, SIZE), Image.BILINEAR)
            return np.asarray(heat, dtype=np.float32) / 255.0

        # 모델이 실제로 본 그림 = 정규화를 되돌린 전처리 결과 (히트맵과 위치가 정확히 맞음)
        mean = torch.tensor(self.processor.image_mean).view(3, 1, 1)
        std = torch.tensor(self.processor.image_std).view(3, 1, 1)
        shown = (pixel_values[0] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        return np.uint8(shown * 255), scores, cam_for


def jet(x: np.ndarray) -> np.ndarray:
    """0~1 값을 파랑→초록→빨강 색으로 바꾼다 (matplotlib 없이)."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def overlay(shown: np.ndarray, cam: np.ndarray) -> np.ndarray:
    return np.uint8(0.55 * shown + 0.45 * jet(cam) * 255)


def font(size=13):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def tile(arr, lines, color=(30, 30, 30)):
    """그림 1장 + 아래 설명 2줄 → 타일 이미지"""
    canvas = Image.new("RGB", (SIZE, SIZE + LABEL_H), "white")
    if arr is None:
        ImageDraw.Draw(canvas).rectangle([0, 0, SIZE, SIZE], fill=(235, 235, 235))
    else:
        canvas.paste(Image.fromarray(arr), (0, 0))
    draw = ImageDraw.Draw(canvas)
    for i, text in enumerate(lines[:2]):
        draw.text((4, SIZE + 2 + i * 16), text, fill=color, font=font())
    return canvas


def load_targets(args):
    """(경로, 정답, 출처) 목록"""
    targets = []
    for ev in args.from_eval or []:
        report = json.loads(Path(ev).read_text(encoding="utf-8"))
        for r in report["misclassified"]:
            path = Path(args.data) / report["split"] / r["true"] / r["file"]
            targets.append((path, r["true"], f"{report['split']}-wrong"))
    for p in args.images or []:
        p = Path(p)
        true = p.parent.name if p.parent.name in ("angular_leaf_spot", "bean_rust", "healthy") else ""
        targets.append((p, true, "manual"))
    return targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/beans_resnet18_full_aug")
    parser.add_argument("--compare-model", default=None, help="같은 사진을 비교할 모델 (예: E0)")
    parser.add_argument("--from-eval", nargs="*", help="evaluate_split.py 결과 JSON (오답 사진 자동 선택)")
    parser.add_argument("--images", nargs="*", help="직접 지정할 사진 경로")
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--out", default="reports/figures/gradcam")
    args = parser.parse_args()

    targets = load_targets(args)
    if not targets:
        raise SystemExit("--from-eval 또는 --images 로 사진을 지정하세요.")

    main_cam = GradCAM(args.model)
    cmp_cam = GradCAM(args.compare_model) if args.compare_model else None
    main_name = Path(args.model).name
    cmp_name = Path(args.compare_model).name if args.compare_model else ""

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, index = [], []
    for path, true, source in targets:
        with Image.open(path) as raw:
            image = raw.convert("RGB")
        shown, scores, cam_for = main_cam.run(image)
        pred = max(scores, key=scores.get)

        tiles = [tile(shown, [f"{path.name[:30]}", f"true: {true or '?'}"])]
        ok = (pred == true)
        tiles.append(tile(overlay(shown, cam_for(pred)),
                          [f"{main_name[:24]}", f"pred {pred} {scores[pred]:.2f}"],
                          (0, 110, 0) if ok else (190, 0, 0)))
        if true and not ok:   # 틀렸다면: 정답 클래스의 근거는 어디였는지도 함께 표시
            tiles.append(tile(overlay(shown, cam_for(true)), ["(true-class evidence)", f"{true} {scores[true]:.2f}"]))
        else:
            tiles.append(tile(None, ["", ""]))
        cmp_pred = ""
        if cmp_cam:
            c_shown, c_scores, c_cam_for = cmp_cam.run(image)
            cmp_pred = max(c_scores, key=c_scores.get)
            tiles.append(tile(overlay(c_shown, c_cam_for(cmp_pred)),
                              [f"{cmp_name[:24]}", f"pred {cmp_pred} {c_scores[cmp_pred]:.2f}"],
                              (0, 110, 0) if cmp_pred == true else (190, 0, 0)))

        row = Image.new("RGB", (SIZE * len(tiles) + 6 * (len(tiles) - 1), SIZE + LABEL_H), "white")
        for i, t in enumerate(tiles):
            row.paste(t, (i * (SIZE + 6), 0))
        row_path = out / f"gradcam_{source}_{path.stem}.png"
        row.save(row_path)
        rows.append(row)
        index.append({"source": source, "image": str(path), "true": true, "pred": pred,
                      "score": round(scores[pred], 4), "compare_pred": cmp_pred, "figure": row_path.name})
        print(f"{source:<11} {path.name:<32} true={true or '?':<18} pred={pred:<18} {scores[pred]:.3f}"
              + (f"  | {cmp_name}: {cmp_pred}" if cmp_cam else ""))

    # 전체 요약 그림 1장 (세로로 이어 붙임)
    width = max(r.width for r in rows)
    summary = Image.new("RGB", (width, sum(r.height + 8 for r in rows)), "white")
    y = 0
    for r in rows:
        summary.paste(r, (0, y))
        y += r.height + 8
    summary.save(out / "gradcam_summary.png")
    with open(out / "gradcam_index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(index[0].keys()))
        writer.writeheader()
        writer.writerows(index)
    print(f"\n저장: {out}/gradcam_*.png ({len(rows)}장), gradcam_summary.png, gradcam_index.csv")


if __name__ == "__main__":
    main()
