# 산출물 3. API 명세서

- Base URL: `http://127.0.0.1:8000` (Docker Compose `web`, 127.0.0.1에만 바인딩)
- 문서: Swagger UI `/docs`, OpenAPI `/openapi.json` (`evidence/05_web_api/web_swagger_docs.png`)
- 응답 예시는 **모두 실제 응답**이다. 점수는 읽기 쉽게 소수 4자리로 반올림했고, 원문은 각 증빙 캡처에 있다.

## 1. 엔드포인트 목록

| API ID | Method | URL | 요청 | 성공 | 주요 응답 필드 | 실패 |
|---|---|---|---|---:|---|---|
| API-01 | GET | `/health` | 없음 | 200 | `status` | 서버 미실행 시 응답 없음 |
| API-02 | POST | `/api/sync/classify/` | `multipart/form-data`, 필드 `image` | 200 | `model_id`, `top1`, `all`, `inference_ms`, `cache_hit` | 400, 413 |
| API-03 | POST | `/api/async/classify/` | `multipart/form-data`, 필드 `image` | **202** | `job_id`, `status`, `check_url` | 400, 413, 503 |
| API-04 | GET | `/api/async/jobs/{job_id}/` | 경로 변수 정수 `job_id` | 200 | `job_id`, `status`, `model_id`, `top1`, `all`, `inference_ms`, `cache_hit`, `error_message` | 404 |
| WEB-01 | GET | `/vision/sync/` | 없음 | 200 | HTML (동기 화면) | — |
| WEB-02 | GET | `/vision/async/` | 없음 | 200 | HTML (비동기 화면) | — |
| (비공개) | GET | `/` | 없음 | 307 | `/vision/sync/`로 이동 (`include_in_schema=False`) | — |

> **상태코드의 의미 구분**
> `202 Accepted` = 작업 **접수**만 완료 (분석 완료 아님) · API-04의 `200` + `"status":"done"` = 분석 **완료**
> 응답 필드명은 `all`이다. `all_scores`는 DB 컬럼명이며 API 응답에 쓰지 않는다.

## 2. 입력 검증 규칙 (서버 코드 기준)

| 검사 | 위치 | 실패 시 |
|---|---|---|
| 5MB 초과 (5×1024×1024 바이트) | `router_sync.py` / `router_async.py` (`MAX_UPLOAD_BYTES + 1`바이트까지만 읽음) | **413** `5MB 이하 파일만 허용합니다.` |
| 빈 파일 | `services.load_rgb_image` | 400 `이미지 파일은 1바이트 이상, 5MB 이하이어야 합니다.` |
| 형식이 JPEG/PNG/WEBP가 아님 | `services.load_rgb_image` (PIL로 실제 디코딩) | 400 `JPG, PNG, WEBP 이미지만 허용합니다.` |
| 1,200만 픽셀 초과 | `services.load_rgb_image` | 400 `이미지 해상도가 너무 큽니다(최대 1200만 픽셀).` |
| 이미지로 열리지 않음 (예: `.txt`) | `services.load_rgb_image` | 400 `올바른 이미지 파일이 아닙니다.` |

- 확장자가 아니라 **파일 내용을 열어서** 검사한다.
- 비동기는 **큐에 넣기 전에** 같은 검사를 한다 → 잘못된 파일로 작업이 생기지 않는다.

## 3. API-02 동기 분류

| 항목 | 내용 |
|---|---|
| 동작 | 캐시 확인 → (MISS면) 추론 → 캐시 저장 → 결과 반환. **응답을 받으면 분석이 끝난 상태** |
| 캐시 키 | `vision:classify:{sha256(MODEL_PATH)[:16]}:{sha256(이미지 바이트)}` · TTL 600초 |
| `inference_ms` | 모델 추론 시간. **첫 요청은 모델 로딩 포함**(cold), 캐시 HIT면 `null`(추론 생략, 0ms가 아님) |
| `score` | softmax 점수. 보정된 확률이 아님 |

**성공 응답 — 캐시 MISS (E3 모델 교체 직후 첫 요청, `bean_rust_val.0.jpg`)**

```json
{
  "model_id": "artifacts/beans_resnet18_full_aug",
  "top1": {"label": "bean_rust", "score": 0.9970},
  "all": [
    {"label": "bean_rust", "score": 0.9970},
    {"label": "angular_leaf_spot", "score": 0.0026},
    {"label": "healthy", "score": 0.0004}
  ],
  "inference_ms": 2990.65,
  "cache_hit": false
}
```
증빙: `evidence/05_web_api/model_switch_E3_api_db.png` (2990.65ms = cold start, 모델 로딩 포함)

**성공 응답 — 같은 이미지 재요청 (캐시 HIT, `healthy_test.20.jpg`, E0 운영 시점)**

```json
{
  "model_id": "artifacts/beans_resnet18",
  "top1": {"label": "healthy", "score": 0.6793},
  "all": [
    {"label": "healthy", "score": 0.6793},
    {"label": "bean_rust", "score": 0.2353},
    {"label": "angular_leaf_spot", "score": 0.0854}
  ],
  "inference_ms": null,
  "cache_hit": true
}
```
증빙: `evidence/07_infra/redis_cache_miss_hit_ttl.png` (1회차 `cache_hit:false, inference_ms:33.59` → 2회차 HIT, Redis TTL 588초 확인)

**오류 응답**

```json
{"detail": "올바른 이미지 파일이 아닙니다."}
```
```json
{"detail": "5MB 이하 파일만 허용합니다."}
```
증빙: `evidence/04_sync_api_01/api_sync_errors.png` (`.txt` → 400, 6MB 랜덤 파일 → 413)

## 4. API-03 비동기 작업 접수

| 항목 | 내용 |
|---|---|
| 동작 | 이미지 검증 → `media/jobs/<uuid>.bin` 저장 → `image_jobs` 행 생성(`queued`) → `classify_job.delay(job_id)` → **202** |
| 응답 시점 | 추론 전 (접수 HTTP 중앙값 9.12ms, [산출물 6](06_성능비교_보고서.md)) |
| 503 | 브로커(Redis) 등록 실패 시. 해당 행은 `failed` + 메시지로 기록됨 |

**성공 응답 (202)**

```json
{"job_id": 8, "status": "queued", "check_url": "/api/async/jobs/8/"}
```
증빙: `evidence/05_web_api/model_switch_E3_api_db.png`, `evidence/05_web_api/api_async_202_done_db.png` (`HTTP/1.1 202 Accepted` 헤더)

**오류 응답**

| 입력 | 상태 | 응답 | 증빙 |
|---|---:|---|---|
| `.txt` 파일 | 400 | `{"detail":"올바른 이미지 파일이 아닙니다."}` | `evidence/05_web_api/api_async_errors.png` |
| 6MB 랜덤 파일 | 413 | `{"detail":"5MB 이하 파일만 허용합니다."}` | `evidence/05_web_api/api_async_errors.png` |
| Redis 중지 상태 | **503** | `{"detail":"작업 큐 등록 실패: Redis 연결을 확인하세요."}` (응답까지 20.67초) | `evidence/08_failure_test/redis_down_async_503.png` |

## 5. API-04 작업 상태 조회

| `status` | 의미 | `top1`, `all` | `error_message` |
|---|---|---|---|
| `queued` | 접수됨, worker가 아직 처리 전 | `null` | `null` |
| `running` | worker 처리 중 (빠른 작업은 거의 관찰되지 않음) | `null` | `null` |
| `done` | 완료 | 결과 | `null` |
| `failed` | 실패 | `null` | 실패 원인 |

**완료 응답 — 캐시 MISS로 worker가 직접 추론 (job 2, E0 운영 시점)**

```json
{
  "job_id": 2,
  "status": "done",
  "model_id": "artifacts/beans_resnet18",
  "top1": {"label": "bean_rust", "score": 0.4388},
  "all": [
    {"label": "bean_rust", "score": 0.4388},
    {"label": "healthy", "score": 0.4356},
    {"label": "angular_leaf_spot", "score": 0.1255}
  ],
  "inference_ms": 2254.13,
  "cache_hit": false,
  "error_message": null
}
```

**대기 응답 — worker 정지 중 (job 3)**

```json
{"job_id": 3, "status": "queued", "model_id": "artifacts/beans_resnet18",
 "top1": null, "all": null, "inference_ms": null, "cache_hit": null, "error_message": null}
```
증빙: `evidence/07_infra/async_worker_stopped_queued.png` → worker 재시작 후 `done` (`evidence/07_infra/async_worker_restarted_done.png`)

**오류 응답 (404)**

```json
{"detail": "해당 작업이 없습니다."}
```
증빙: `evidence/05_web_api/api_async_errors.png` (`/api/async/jobs/999999/`)

## 6. 실행 근거표

| 검증 ID | 테스트 입력 | 기대 | 실제 | 증빙 |
|---|---|---|---|---|
| API-01-T01 | `GET /health` | 200 `{"status":"ok"}` | 200 `{"status":"ok"}` (content-length 15) | `health.png` |
| API-02-T01 | 콩잎 3클래스 각 1장 (01, E0) | 200 / 3클래스 라벨 | 200 / 3장 모두 정답 | `api_sync_after_*.png` |
| API-02-T02 | 같은 이미지 2회 | MISS → HIT | `false` → `true`, `inference_ms` 33.59 → `null` | `redis_cache_miss_hit_ttl.png` |
| API-02-T03 | `.txt` | 400 | 400 | `api_sync_errors.png` |
| API-02-T04 | 6MB 파일 | 413 | 413 | `api_sync_errors.png` |
| API-02-T05 | Redis 중지 상태 | 캐시 없이 200 | 200, `cache_hit:false`, 응답 8.06초 | `redis_down_sync_200.png` |
| API-03-T01 | 콩잎 이미지 | 202 / job_id | 202 / job_id 1 | `api_async_202_done_db.png` |
| API-03-T02 | `.txt` | 400 | 400 | `api_async_errors.png` |
| API-03-T03 | 6MB 파일 | 413 | 413 | `api_async_errors.png` |
| API-03-T04 | Redis 중지 상태 | 503 | 503 | `redis_down_async_503.png` |
| API-04-T01 | 접수된 job 1 | 200 / done | 200 / done, `cache_hit:true` | `api_async_202_done_db.png` |
| API-04-T02 | worker 정지 중 job 3 | queued 유지 | queued → 재시작 후 done | `async_worker_stopped_queued.png`, `async_worker_restarted_done.png` |
| API-04-T03 | `job_id=999999` | 404 | 404 | `api_async_errors.png` |
| WEB-01/02 | 화면 접속·업로드 | 200 / 결과 표시 | 동기 MISS·HIT, 비동기 queued·done 표시 | `web_sync_*.png`, `web_async_*.png` |

## 7. 사용 예 (curl)

```bash
IMG=dataset/test/healthy/healthy_test.16.jpg
curl -s http://127.0.0.1:8000/health ; echo
curl -s -X POST -F "image=@${IMG}" http://127.0.0.1:8000/api/sync/classify/ ; echo
curl -i -X POST -F "image=@${IMG}" http://127.0.0.1:8000/api/async/classify/ ; echo
curl -s http://127.0.0.1:8000/api/async/jobs/1/ ; echo     # 반환된 job_id로 조회
```
