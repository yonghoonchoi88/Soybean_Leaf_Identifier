<div align="center">

# Soybean Leaf Disease Classification · From Fine-tuning to Sync/Async Serving

</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11"> <img src="https://img.shields.io/badge/PyTorch-2.15%20train%20%7C%202.5.1%20serve-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.15 train | 2.5.1 serve"> <img src="https://img.shields.io/badge/Transformers-ResNet--18-FFD21E?logo=huggingface&logoColor=black" alt="Transformers ResNet-18"> <img src="https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white" alt="FastAPI 0.115+"> <img src="https://img.shields.io/badge/Celery-5.4%2B-37814A?logo=celery&logoColor=white" alt="Celery 5.4+"> <img src="https://img.shields.io/badge/Redis-7.4-DC382D?logo=redis&logoColor=white" alt="Redis 7.4"><br>
  <img src="https://img.shields.io/badge/PostgreSQL-17-4169E1?logo=postgresql&logoColor=white" alt="PostgreSQL 17"> <img src="https://img.shields.io/badge/Docker%20Compose-4%20services-2496ED?logo=docker&logoColor=white" alt="Docker Compose 4 services"> <img src="https://img.shields.io/badge/OpenCV-HSV%20analysis-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV HSV analysis"> <img src="https://img.shields.io/badge/NVIDIA-RTX%203060-76B900?logo=nvidia&logoColor=white" alt="NVIDIA RTX 3060"> <img src="https://img.shields.io/badge/Ubuntu-22.04%20WSL2-E95420?logo=ubuntu&logoColor=white" alt="Ubuntu 22.04 WSL2">
</p>

[한국어](README.md) | **English**

Using 1,295 soybean leaf photos, we fine-tuned ResNet-18 into a 3-class classifier (**angular leaf spot · bean rust · healthy**) and served it with Docker Compose: a sync API, a Redis cache, Celery async jobs, and PostgreSQL job records.
We diagnosed why the baseline fell short and improved it in two steps, **the course guide's Full Fine-tuning setting → adding augmentation**. The model was **selected on Validation and evaluated once on Test**.

> 📑 The deliverable reports (01–09) are written in Korean. The tables and figures carry the key numbers.

## At a Glance

<p align="center">
  <img src="reports/figures/pipeline.en.png" alt="Pipeline" width="100%">
</p>

<table>
  <tr>
    <th width="10%"></th>
    <th width="45%">① Model</th>
    <th width="45%">② Service</th>
  </tr>
  <tr>
    <td align="center"><b>What we did</b></td>
    <td>Data check → Head-only baseline → Full Fine-tuning · augmentation experiments → selection on Val → Test once → error analysis</td>
    <td>Sync API + Redis cache · Celery async · PostgreSQL records · model swap · failure tests</td>
  </tr>
  <tr>
    <td align="center"><b>Key numbers</b></td>
    <td>Val <b>0.767 → 0.993</b> · final Test <b>0.953</b> · missed diseased leaves Val <b>11 → 1</b>, Test <b>0</b></td>
    <td>Cache HIT <b>−91%</b> · async accept <b>9.1 ms</b> · live tests <b>27 / 27 PASS</b></td>
  </tr>
  <tr>
    <td align="center"><b>Records</b></td>
    <td><a href="reports/01_FineTuning_결과보고서.md">Report 1</a> · <a href="reports/08_최종기술보고서.md">Report 8</a></td>
    <td><a href="#deliverables">Reports 2–7</a> · <a href="reports/07_기능테스트_오류분석.md">Function tests</a></td>
  </tr>
  <tr>
    <td align="center"><b>Overview</b></td>
    <td align="center" valign="top"><img src="reports/figures/dashboard.en.png" alt="Model results at a glance"><br>Model selection · final evaluation · error types</td>
    <td align="center" valign="top"><img src="reports/figures/service.en.png" alt="Service verification at a glance"><br>Web pages · response time · API and failure tests</td>
  </tr>
</table>

### Experiment Setup — What We Changed

| Run | What changed | Trained part | Learning rate | Train augmentation | Validation |
|---|---|---|---:|---|---:|
| **E1** Head-only (required) | Provided command as is (`--freeze-backbone`) | Final classification head only (1,539 params) | 0.0002 | None (the provided code has no augmentation) | 0.767 |
| **E2** Full Fine-tuning | Unfreeze + ¼ learning rate (guide 6-1 B setting) | **Everything incl. backbone** (about 11.18M params) | 0.00005 | None (left out to isolate the Full effect) | 0.970 |
| **E3** Full + Aug ⭐ | E2 plus **Train augmentation only** | Everything | 0.00005 | Horizontal flip (50%) · rotation ±20° · brightness/contrast ±20% | **0.993** |

- **Full Fine-tuning:** E1 keeps the "eyes" learned on ImageNet (the backbone) frozen and trains only the last layer. E2 retrains the backbone for soybean lesions too, with a lower learning rate so the pretrained features are not destroyed.
- **Augmentation:** each time a Train photo is read, it is randomly flipped, rotated, and re-lit, making the model robust to shooting angle and lighting. Bean rust (reddish brown) and angular leaf spot (yellowish brown) are told apart by color, so **hue and saturation are not changed.** Validation and Test are never augmented.
- Common: epochs 3 · batch 8 · same Train/Validation · `training/finetune.py` unmodified (E3 uses the wrapper `training/finetune_aug.py`)
- Run IDs: there are **three runs, E1 → E2 → E3**. The baseline was called **E0** during the work and renamed E1 when the documents were cleaned up. The numbers are unchanged. Evaluation screenshots from that time (`eval_val_E1.png` etc.) still show `--tag E0_headonly` and the saved file name `eval_val_E0_headonly.json` (the current file name is `E1_headonly`). The `e3` in training log names (`train_full_e3.log` etc.) means **epochs 3**, not run E3.

## Deliverables

Reports are in Korean.

| # | Document | Key content |
|---:|---|---|
| 1 | [Fine-tuning results](reports/01_FineTuning_결과보고서.md) | Data check · E1/E2/E3 · selection on Val · Test once · model swap |
| 2 | [System architecture](reports/02_시스템_아키텍처.md) | Docker Compose, 4 services · roles of Redis DB1/2/3 |
| 3 | [API specification](reports/03_API_명세서.md) | 6 APIs · real response JSON · 400/413/404/503 |
| 4 | [DB schema · job history](reports/04_DB_구조_작업이력.md) | DDL · state transitions · jobs 1\~21 (including `failed`) |
| 5 | [AI button processing flow](reports/05_AI버튼_처리흐름.md) | Sync and async flow diagrams (files · functions · line numbers) · code map |
| 6 | [Performance comparison](reports/06_성능비교_보고서.md) | `timings.csv` · reading the cache and async results |
| 7 | [Function tests · errors](reports/07_기능테스트_오류분석.md) | 27 live tests · 2 mock tests · 12 errors |
| 8 | [**Final technical report**](reports/08_최종기술보고서.md) | Summary · improvement techniques ①\~④ · limitations · proposals |
| 9 | [Performance analysis · improvement](reports/09_성능분석_개선보고서.md) | Problems P1\~P5 → causes → fixes → re-verification |

<details>
<summary><b>Quick start</b> — place data → train → <code>docker compose up</code> (data, models and <code>.env</code> are not in the repo)</summary>

```bash
# 1. Environment (WSL2 Ubuntu 22.04 · Python 3.11 · Docker Desktop WSL Integration)
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Data: keep the original zips; place them as dataset/train · val · test, which the provided code reads
mkdir -p dataset
unzip -q ~/beans_source/train.zip -d dataset/
unzip -q ~/beans_source/validation.zip -d dataset/ && mv dataset/validation dataset/val
unzip -q ~/beans_source/test.zip -d dataset/
python scripts/check_dataset.py                          # 1034 / 133 / 128, 0 duplicates or leaks

# 3. Training (GPU) — E3 is the served model
python -m training.finetune     --data dataset --output artifacts/beans_resnet18          --epochs 3 --batch-size 8 --freeze-backbone
python -m training.finetune     --data dataset --output artifacts/beans_resnet18_full     --epochs 3 --batch-size 8 --lr 0.00005
python -m training.finetune_aug --data dataset --output artifacts/beans_resnet18_full_aug --epochs 3 --batch-size 8 --lr 0.00005

# 4. Service
cp .env.example .env
sed -i 's#^MODEL_PATH=.*#MODEL_PATH=artifacts/beans_resnet18_full_aug#' .env
mkdir -p artifacts media/jobs
docker compose up -d --build
curl -s http://127.0.0.1:8000/health ; echo             # pages: /vision/sync/ · /vision/async/ · /docs
# check: /health → {"status":"ok"} · sync → top1·cache_hit · async → 202·job_id → done · a row in DB image_jobs
# switch to the guide's default path (E1): sed -i 's#^MODEL_PATH=.*#MODEL_PATH=artifacts/beans_resnet18#' .env && docker compose up -d --force-recreate web worker
docker compose stop                                      # stop (do not use down -v; it deletes the DB and cache)

# 5. Tests · analysis
python -m pytest -q tests/test_api.py                    # mock tests: 4 passed
python scripts/measure_timings.py --repeat 5             # with the server running → reports/data/timings.csv
uv pip install opencv-python-headless matplotlib         # for analysis and figures only
python scripts/hsv_rule_analysis.py && python scripts/make_figures.py
```

</details>

<details>
<summary><b>Added tools</b> — 7 files added; the provided code is not modified</summary>

| File | What it does |
|---|---|
| `scripts/check_dataset.py` | Per-class counts, SHA-256 duplicate / leak / label-conflict check |
| `training/finetune_aug.py` | Reuses the provided `finetune.py` and augments Train only (E3) |
| `scripts/evaluate_split.py` | Accuracy · confusion matrix · P/R/F1 · score-margin bins · list of errors |
| `scripts/measure_timings.py` | Repeated sync/async × MISS/HIT timing, warm-up kept separate |
| `scripts/gradcam.py` | Grad-CAM on misclassified photos, comparison across models |
| `scripts/hsv_rule_analysis.py` | OpenCV HSV lesion-candidate analysis + auxiliary rule simulation |
| `scripts/make_figures.py` | Reads only the raw files in `reports/data` to draw the dashboards and charts (Korean and `.en` English) |

</details>

<details>
<summary><b>Folder structure</b></summary>

```text
├── ai/ · main.py          # provided code: model · service · routers · cache · Celery · DB
├── training/              # provided finetune.py + added finetune_aug.py
├── scripts/               # added tools (check · evaluate · measure · analyze · draw)
├── templates/ · static/   # provided web pages
├── tests/                 # test_api.py (02 mock tests) · legacy_01/ (01-stage records)
├── compose.yaml · Dockerfile · requirements.txt · .env.example
├── reports/
│   ├── 01_*.md ~ 09_*.md  # deliverables (Korean)
│   ├── data/              # raw results (training reports · evaluations · timings · HSV)
│   ├── figures/           # pipeline · model/service boards (*.en.png = English) · charts · diagrams · gradcam · hsv
│   └── evidence/          # run screenshots 00_env ~ 08_failure_test (by stage)
└── artifacts/ · media/    # models · uploaded images (excluded from Git, folders kept)
```

</details>

<details>
<summary><b>Environment · security · limitations</b></summary>

- **Environment:** WSL2 Ubuntu 22.04.5 · Python 3.11.16 · transformers 4.57.6 · RTX 3060 (training) · Docker 29.9.2 / Compose v5.6.1 · inference on CPU in containers
- **Security:** `.env` · `dataset/` · `artifacts/` · `media/jobs/` are excluded from Git. The DB password in `compose.yaml` is the course's local practice value (ports bound to `127.0.0.1` only) → move it to `.env` for production
- **Limitations:**
  1. All 6 Test errors are bean rust ↔ angular leaf spot confusions (bean rust Recall 0.88) → [Report 9](reports/09_성능분석_개선보고서.md)
  2. Single-source data — on internet soybean photos the model scored bean_rust 0.414 vs healthy 0.410, effectively no decision; non-leaf photos are also forced into one of the 3 classes
  3. One run per experiment (seed 42) — E2 → E3 differs by 3 Val images, so the augmentation effect is not confirmed
  4. With Redis stopped, sync took 8.06 s to return 200 and async 20.67 s to return 503 · concurrent load not measured
- **Sources:** data [AI-Lab-Makerere/beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans) · model [microsoft/resnet-18](https://huggingface.co/microsoft/resnet-18)

</details>
