<div align="center">

# 콩잎 질병 분류 · Fine-tuning부터 동기·비동기 서빙까지

**ResNet-18 · FastAPI · Redis · Celery · PostgreSQL · Docker Compose**

</div>

콩잎 사진 1,295장으로 ResNet-18을 **반점병 · 녹병 · 정상** 3클래스로 Fine-tuning하고, 동기 API · Redis 캐시 · Celery 비동기 작업 · PostgreSQL 작업 기록을 Docker Compose로 서비스했습니다.
기준선의 원인을 진단해 **한 번에 한 조건만 바꿔** 개선했고, 모델은 **Validation으로 고른 뒤 Test에 한 번** 평가했습니다.

<p align="center">
  <img src="reports/figures/dashboard.png" alt="프로젝트 한눈에 보기" width="100%">
</p>

| | 모델 | 서비스 |
|---|---|---|
| **한 일** | 데이터 점검 → Head-only 기준선 → Full Fine-tuning · 증강 실험 → Val 선택 → Test 1회 → 오답 분석 | 동기 API + Redis 캐시 · Celery 비동기 · PostgreSQL 기록 · 모델 교체 · 장애 테스트 |
| **핵심 숫자** | Val **0.767 → 0.993** · 최종 Test **0.953** · 병든 잎 놓침 Val **11 → 1**, Test **0** | 캐시 HIT **−91%** · 비동기 접수 **9.1 ms** · 실환경 테스트 **27 / 27 PASS** |
| **정직하게 남긴 것** | 증강 효과 크기는 미확정 · 색 규칙은 기각 · [Test 사용 이력 공개](reports/01_FineTuning_결과보고서.md#부록-test-사용-이력-원칙-점검) | Redis 장애 시 응답 8초·20초 지연 → 개선안으로 이관 |

## 산출물

| # | 문서 | 핵심 |
|---:|---|---|
| 1 | [Fine-tuning 결과보고서](reports/01_FineTuning_결과보고서.md) | 데이터 점검 · E0/E2/E3 · Val 선택 · Test 1회 · 모델 교체 |
| 2 | [시스템 아키텍처](reports/02_시스템_아키텍처.md) | Docker Compose 4서비스 · Redis DB1/2/3 역할 |
| 3 | [API 명세서](reports/03_API_명세서.md) | 6개 API · 실제 응답 JSON · 400/413/404/503 |
| 4 | [DB 구조 · 작업이력](reports/04_DB_구조_작업이력.md) | DDL · 상태 전이 · job 1~21 (`failed` 포함) |
| 5 | [AI 버튼 처리흐름](reports/05_AI버튼_처리흐름.md) | 동기·비동기 시퀀스도 · 코드 연결표 |
| 6 | [성능 비교](reports/06_성능비교_보고서.md) | `timings.csv` · 캐시·비동기 해석 |
| 7 | [기능 테스트 · 오류](reports/07_기능테스트_오류분석.md) | 실환경 27건 · 모의 2건 · 오류 12건 |
| 8 | [**최종 기술 보고서**](reports/08_최종기술보고서.md) | 종합 · 개선 기술 ①~④ · 한계 · 개선 제안 |

<details>
<summary><b>빠른 실행</b> — 데이터 배치 → 학습 → <code>docker compose up</code> (데이터·모델·<code>.env</code>는 저장소에 없음)</summary>

```bash
# 1. 환경 (WSL2 Ubuntu 22.04 · Python 3.11 · Docker Desktop WSL Integration)
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt

# 2. 데이터: 원본 zip은 보존, 제공 코드가 읽는 dataset/train · val · test 로 배치
mkdir -p dataset
unzip -q ~/beans_source/train.zip -d dataset/
unzip -q ~/beans_source/validation.zip -d dataset/ && mv dataset/validation dataset/val
unzip -q ~/beans_source/test.zip -d dataset/
python scripts/check_dataset.py                          # 1034 / 133 / 128, 중복·누출 0

# 3. 학습 (GPU) — E3가 서비스 모델
python -m training.finetune     --data dataset --output artifacts/beans_resnet18          --epochs 3 --batch-size 8 --freeze-backbone
python -m training.finetune     --data dataset --output artifacts/beans_resnet18_full     --epochs 3 --batch-size 8 --lr 0.00005
python -m training.finetune_aug --data dataset --output artifacts/beans_resnet18_full_aug --epochs 3 --batch-size 8 --lr 0.00005

# 4. 서비스
cp .env.example .env
sed -i 's#^MODEL_PATH=.*#MODEL_PATH=artifacts/beans_resnet18_full_aug#' .env
mkdir -p artifacts media/jobs
docker compose up -d --build
curl -s http://127.0.0.1:8000/health ; echo             # 화면: /vision/sync/ · /vision/async/ · /docs
# 확인: /health → {"status":"ok"} · 동기 → top1·cache_hit · 비동기 → 202·job_id → done · DB image_jobs 행
# 가이드 기본 경로(E0)로 전환: sed -i 's#^MODEL_PATH=.*#MODEL_PATH=artifacts/beans_resnet18#' .env && docker compose up -d --force-recreate web worker
docker compose stop                                      # 정지 (down -v 는 DB·캐시를 지우므로 사용 안 함)

# 5. 테스트 · 분석
python -m pytest -q tests/test_api.py                    # 모의 테스트 4 passed
python scripts/measure_timings.py --repeat 5             # 서버 실행 중 → reports/data/timings.csv
uv pip install opencv-python-headless matplotlib         # 분석·그림 전용
python scripts/hsv_rule_analysis.py && python scripts/make_figures.py
```

</details>

<details>
<summary><b>추가한 도구</b> — 제공 코드는 수정하지 않고 7개 파일 추가</summary>

| 파일 | 하는 일 |
|---|---|
| `scripts/check_dataset.py` | 클래스별 개수, SHA-256 중복·누출·라벨 충돌 점검 |
| `training/finetune_aug.py` | 제공 `finetune.py`를 재사용하며 Train에만 증강 적용 (E3) |
| `scripts/evaluate_split.py` | 정확도·혼동행렬·P/R/F1·점수 차 구간·오답 목록 |
| `scripts/measure_timings.py` | 동기/비동기 × MISS/HIT 반복 측정, 워밍업 분리 |
| `scripts/gradcam.py` | 오답 사진 Grad-CAM, 모델 간 비교 |
| `scripts/hsv_rule_analysis.py` | OpenCV HSV 병반 후보 분석 + 보조 규칙 시뮬레이션 |
| `scripts/make_figures.py` | `reports/data` 원본만 읽어 대시보드·그래프 생성 |

</details>

<details>
<summary><b>폴더 구조</b></summary>

```text
├── ai/ · main.py          # 제공 코드: 모델·서비스·라우터·캐시·Celery·DB
├── training/              # 제공 finetune.py + 추가 finetune_aug.py
├── scripts/               # 추가 도구 (점검·평가·측정·분석·그림)
├── templates/ · static/   # 제공 웹 화면
├── tests/                 # test_api.py (02 모의 테스트) · legacy_01/ (01 단계 기록)
├── compose.yaml · Dockerfile · requirements.txt · .env.example
├── reports/
│   ├── 01_*.md ~ 08_*.md  # 산출물
│   ├── data/              # 원본 결과 (학습 리포트·평가·응답 시간·HSV)
│   ├── figures/           # 대시보드·그래프·구성도·gradcam·hsv
│   └── evidence/          # 실행 캡처 00_env ~ 08_failure_test (단계별)
└── artifacts/ · media/    # 모델·업로드 이미지 (Git 제외, 폴더만 유지)
```

</details>

<details>
<summary><b>환경 · 보안 · 한계</b></summary>

- **환경:** WSL2 Ubuntu 22.04.5 · Python 3.11.16 · transformers 4.57.6 · RTX 3060(학습) · Docker 29.9.2 / Compose v5.6.1 · 컨테이너 추론 CPU
- **보안:** `.env` · `dataset/` · `artifacts/` · `media/jobs/`는 Git 제외. `compose.yaml`의 DB 비밀번호는 교재 제공 로컬 실습값(포트는 `127.0.0.1`에만 바인딩) → 운영 시 `.env` 분리 권장
- **한계:** 단일 출처 데이터 · 실험당 1회 실행 · 동시 부하 미측정 · 학습 범위 밖 입력도 3클래스 중 하나로 분류
- **출처:** 데이터 [AI-Lab-Makerere/beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans) · 모델 [microsoft/resnet-18](https://huggingface.co/microsoft/resnet-18)

</details>
