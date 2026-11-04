"""[개선 실험 ①] 이미지 증강(Data Augmentation)을 추가해 학습하는 래퍼 스크립트.

- 제공된 training/finetune.py 는 한 줄도 수정하지 않는다.
- finetune.py 의 학습 루프·시드·저장 방식을 그대로 재사용하고,
  "이미지를 읽는 부분(FolderImages)"만 증강 버전으로 바꿔 끼운다.
- 증강은 Train 이미지에만 적용한다. Validation 에 증강을 넣으면 채점 기준이 흔들리므로 절대 적용하지 않는다.

사용 예 (E2 Full Fine-tuning 과 같은 조건 + 증강만 추가 → 증강 효과만 비교 가능):
  python -u -m training.finetune_aug --data dataset --output artifacts/beans_resnet18_full_aug \
      --epochs 5 --batch-size 8 --lr 0.00005
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

import training.finetune as ft  # 제공 학습 코드 (수정하지 않음)

# 콩잎 사진에 "현실에서 일어날 법한" 변화만 준다.
# - 좌우 반전·회전: 잎을 어느 방향에서 찍어도 병은 같다
# - 밝기·대비: 햇빛·그늘 등 촬영 조명 차이
# - 색조(hue)·채도는 바꾸지 않는다: 녹병(갈색)·반점병(황갈색)은 "색"이 핵심 단서라서,
#   색을 바꾸면 정답 라벨과 맞지 않는 이미지를 만들 위험이 있다.
AUGMENT = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),               # ±20도 회전 (모서리는 검은색으로 채워짐)
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])
AUGMENT_DESC = ["RandomHorizontalFlip(p=0.5)", "RandomRotation(±20°)",
                "ColorJitter(brightness=0.2, contrast=0.2)", "hue/saturation 미변경"]


class AugFolderImages(ft.FolderImages):
    """제공 FolderImages 를 상속: 파일 목록 수집은 그대로, 이미지 읽을 때만 증강을 끼워 넣는다."""

    def __init__(self, root, classes, processor):
        super().__init__(root, classes, processor)
        self.augment = Path(root).name == "train"   # train 폴더일 때만 True, val 은 False

    def __getitem__(self, index):
        path, label = self.records[index]
        with Image.open(path) as raw:
            image = raw.convert("RGB")
        if self.augment:
            image = AUGMENT(image)                    # 매 epoch마다 무작위로 조금씩 다른 이미지가 됨
        tensor = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return {"pixel_values": tensor, "labels": torch.tensor(label, dtype=torch.long)}


# ft.run() 은 실행 시점에 모듈 안의 FolderImages 라는 "이름"을 찾아서 쓴다.
# 그 이름이 가리키는 클래스만 바꿔 주면, 학습 코드를 고치지 않고도 증강 데이터셋이 사용된다.
ft.FolderImages = AugFolderImages


if __name__ == "__main__":
    # finetune.py 와 똑같은 실행 옵션
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--output", default="artifacts/beans_resnet18_full_aug")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()

    print(f"[증강 실험] train 에만 적용: {AUGMENT_DESC}")
    ft.run(args)

    # 제공 코드의 training_report.json 에는 실행 옵션이 없으므로, 실험 재현용 정보를 덧붙인다.
    report_path = Path(args.output) / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["experiment"] = {
        "script": "training/finetune_aug.py (제공 finetune.py 재사용, 미수정)",
        "augmentation_train_only": AUGMENT_DESC,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "freeze_backbone": args.freeze_backbone,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"실험 설정 기록: {report_path}")
