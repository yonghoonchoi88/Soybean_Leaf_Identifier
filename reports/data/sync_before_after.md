# 01 동기 API: 원본 vs Head-only 학습 모델 (2026-11-04)

- 실행환경: WSL2 Ubuntu-22.04 / Python 3.11 (uv) / CPU 추론 (device=-1)
- 원본: microsoft/resnet-18 → 학습: artifacts/beans_resnet18 (Head-only, best val acc 0.7669, epoch 2)
- 입력: 각 클래스 test 첫 이미지 (*_test.0.jpg)

| 정답 | 원본 top1 (score) | 학습 모델 top1 (score) | 2위 (score) | 판정 |
|---|---|---|---|---|
| angular_leaf_spot | green mamba (0.318) | angular_leaf_spot (0.462) | bean_rust (0.356) | 정답 |
| bean_rust | cucumber (0.307) | bean_rust (0.605) | healthy (0.261) | 정답 |
| healthy | bonnet (0.703) | healthy (0.637) | bean_rust (0.248) | 정답 |

| 모델 | Cold inference_ms (서버 재시작 후 첫 요청) | Warm inference_ms |
|---|---:|---:|
| 원본 | 4651.67 | 26.27 / 21.2 |
| 학습 모델 | 2419.73 | 23.76 / 21.74 |

- 주의: 3장 결과는 서비스 연결 시연이며 정확도 지표 아님 (모델 성능 근거는 Val Acc 0.7669)
- 증빙: evidence/04_sync_api_01/api_sync_before_*.png, evidence/04_sync_api_01/api_sync_after_*.png
