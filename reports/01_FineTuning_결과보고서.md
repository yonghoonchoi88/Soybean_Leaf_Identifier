# ResNet18 콩잎 3클래스 Fine-tuning 결과보고서

- 평가 대상: 콩잎 질병 이미지 3종 분류 (angular_leaf_spot / bean_rust / healthy)
- 기준 모델: microsoft/resnet-18 (ImageNet-1K 사전학습, Apache-2.0)
- 학습 방식: **Head-only Fine-tuning** (Backbone freeze, 분류 헤드만 학습)
- 입력 데이터: dataset/train, dataset/val
- 최종 보관 데이터: dataset/test (학습·모델 선택에 사용하지 않음)
- 모델 저장 위치: artifacts/beans_resnet18/
- 실행 일시: 2026-11-04

## 1. 데이터 구성

| 클래스 | Train | Validation | Test |
|---|---:|---:|---:|
| angular_leaf_spot | 345 | 44 | 43 |
| bean_rust | 348 | 45 | 43 |
| healthy | 341 | 44 | 42 |
| **합계** | **1,034** | **133** | **128** |

- 원본 보관: `~/beans_source/{train,validation,test}.zip` (Hugging Face `AI-Lab-Makerere/beans`, MIT)
- 코드 적용 매핑: train → `dataset/train`, **validation → `dataset/val`**, test → `dataset/test`
- 클래스 비율: 세 클래스 약 1:1:1 (Accuracy 해석 시 불균형 영향 적음)
- 데이터 누출 점검: `scripts/check_dataset.py`로 전체 이미지 SHA-256 해시 비교
  - 중복 그룹 0 / Split 간 누출 0 / 라벨 충돌 0
  - 원본 기록: `reports/data/dataset_check.json`
  - 한계: 바이트 단위 완전 일치만 검사, 유사 이미지(같은 잎 재촬영 등)는 미확인

## 2. 실행 설정

| 파라미터 | 값 |
|---|---|
| base_model | microsoft/resnet-18 |
| 출력층 | 1,000 → **3** (classifier.1 재초기화, 학습 로그 경고로 확인) |
| epochs | 3 |
| batch_size | 8 |
| learning_rate | 0.0002 (제공 코드 기본값) |
| freeze_backbone | True |
| seed | 42 (제공 코드) |
| output | artifacts/beans_resnet18 |
| 학습 장치 | GPU |
| 학습 소요 시간 | 28.1초 (`time` real) |

실행 명령 (제공 옵션 그대로, 로그 저장만 추가):

    time python -u -m training.finetune --data dataset --output artifacts/beans_resnet18 \
      --epochs 3 --batch-size 8 --freeze-backbone 2>&1 | tee reports/evidence/train_headonly.log

## 3. 학습 결과

| Epoch | Train Loss | Validation Accuracy | 정답 수 (/133) |
|---:|---:|---:|---:|
| 1 | 0.9924 | 0.6617 | 88 |
| 2 | 0.8108 | **0.7669** | 102 |
| 3 | 0.7078 | 0.7669 | 102 |

- Best Validation Accuracy: **0.7669 (Epoch 2)**
- 저장된 모델: Epoch 2 체크포인트 (Epoch 3은 동일 정확도로 저장 조건 `acc > best_acc` 미충족)
- 해석: Loss가 계속 감소 중이므로 추가 학습 여지 있음 (3 epoch에서 종료)
- Test Accuracy: **미측정** (제공 코드에 Test 평가 절차 없음)

## 4. 생성 파일 확인

- [x] config.json
- [x] model.safetensors
- [x] preprocessor_config.json
- [x] training_report.json

## 5. 서비스 연결 확인 (동기 API)

`.env`의 `MODEL_PATH`를 `microsoft/resnet-18` → `artifacts/beans_resnet18`로 변경 후 서버 재시작, 같은 이미지로 비교:

| 정답 | 원본 모델 top1 (score) | 학습 모델 top1 (score) | 판정 |
|---|---|---|---|
| angular_leaf_spot | green mamba (0.318) | angular_leaf_spot (0.462) | 정답 |
| bean_rust | cucumber (0.307) | bean_rust (0.605) | 정답 |
| healthy | bonnet (0.703) | healthy (0.637) | 정답 |

- 응답 `model_id`가 `artifacts/beans_resnet18`로 바뀐 것 확인
- 3장 결과는 **서비스 연결 시연**이며 정확도 지표가 아님 (모델 성능 근거는 Val Acc 0.7669)
- 상세: `reports/data/sync_before_after.md`

## 6. 결과 판정

- 완료: Head-only Fine-tuning 학습, 모델 저장, 동기 API 연결
- 한계:
  - Test 정량평가 미수행
  - angular_leaf_spot 예시에서 1·2위 점수 차가 작음 (0.46 vs 0.36) → 병반 질병 간 구분 신뢰도 낮을 가능성
  - score는 보정된 확률이 아님 (원본 모델은 healthy 잎을 0.70으로 "bonnet"이라 오답)
- 증빙: `evidence/data_check_result.png`, `evidence/train_headonly_terminal.png`, `evidence/train_headonly_time.png`, `evidence/train_artifacts.png`, `evidence/train_headonly.log`, `evidence/api_sync_before_*.png`, `evidence/api_sync_after_*.png`