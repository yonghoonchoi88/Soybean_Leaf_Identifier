"""동기·비동기 / 캐시 MISS·HIT 응답시간을 같은 조건으로 반복 측정해 timings.csv 로 저장하는 별도 도구.
(제공 코드는 수정하지 않음. 실행 중인 서버에 HTTP 요청만 보낸다.)

측정 규칙 (평가 가이드 산출물 6 기준)
- 워밍업: 모델 첫 로딩이 섞이지 않도록 web·worker 각각 1회씩 먼저 요청하고, 비교에서 제외해 따로 저장
- MISS 그룹: 한 번도 보내지 않은 서로 다른 이미지 N장
- HIT 그룹 : 방금 MISS 그룹에서 보낸 "같은 이미지"를 바로 다시 보냄 (Redis TTL 600초 안)
- 동기  http_ms  = POST 응답까지 시간 (= 결과 확인 시간 final_ms)
- 비동기 http_ms  = POST 202 접수 응답까지 시간
         final_ms = POST 시작 ~ 상태 조회가 done 이 될 때까지 시간 (조회 간격 --poll 초)
- 응답의 실제 cache_hit 값으로 cache_status 를 기록하고, 기대와 다르면 경고

사용 예:
  python scripts/measure_timings.py --repeat 5
"""
import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import httpx

EXTS = {".jpg", ".jpeg", ".png", ".webp"}
FIELDS = ["run_id", "path", "cache_status", "http_ms", "final_ms", "inference_ms", "job_id",
          "http_status", "image", "model_id"]


def pick_images(test_dir: Path, count: int, start: int):
    """클래스를 번갈아 가며 서로 다른 이미지 count장을 고른다 (각 클래스에서 start번째 파일부터)."""
    per_class = []
    for cls in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        files = sorted(f for f in cls.iterdir() if f.is_file() and f.suffix.lower() in EXTS)
        per_class.append(files[start:])
    picked, i = [], 0
    while len(picked) < count:
        group = per_class[i % len(per_class)]
        idx = i // len(per_class)
        if idx >= len(group):
            raise SystemExit("이미지가 부족합니다. --start 값을 줄이세요.")
        picked.append(group[idx])
        i += 1
    return picked


def post(client, url, path):
    data = path.read_bytes()
    t0 = time.perf_counter()
    res = client.post(url, files={"image": (path.name, data, "image/jpeg")})
    return res, (time.perf_counter() - t0) * 1000, t0


def run_sync(client, base, path, run_id, expect):
    res, ms, _ = post(client, f"{base}/api/sync/classify/", path)
    body = res.json()
    status = "HIT" if body.get("cache_hit") else "MISS"
    if expect and status != expect:
        print(f"  ⚠️ 기대 {expect} / 실제 {status}: {path.name}")
    return {"run_id": run_id, "path": "sync", "cache_status": status,
            "http_ms": round(ms, 2), "final_ms": round(ms, 2),
            "inference_ms": body.get("inference_ms"), "job_id": "",
            "http_status": res.status_code, "image": path.name, "model_id": body.get("model_id")}


def run_async(client, base, path, run_id, expect, poll, timeout):
    res, http_ms, t0 = post(client, f"{base}/api/async/classify/", path)
    body = res.json()
    job_id, check_url = body.get("job_id"), body.get("check_url")
    job = {}
    while time.perf_counter() - t0 < timeout:
        time.sleep(poll)
        job = client.get(f"{base}{check_url}").json()
        if job.get("status") in ("done", "failed"):
            break
    final_ms = (time.perf_counter() - t0) * 1000
    status = "HIT" if job.get("cache_hit") else "MISS"
    if job.get("status") != "done":
        print(f"  ⚠️ job {job_id} 상태 {job.get('status')} (완료되지 않음)")
    if expect and status != expect:
        print(f"  ⚠️ 기대 {expect} / 실제 {status}: {path.name}")
    return {"run_id": run_id, "path": "async", "cache_status": status,
            "http_ms": round(http_ms, 2), "final_ms": round(final_ms, 2),
            "inference_ms": job.get("inference_ms"), "job_id": job_id,
            "http_status": res.status_code, "image": path.name, "model_id": job.get("model_id")}


def median(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.median(values), 2) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--data", default="dataset/test")
    parser.add_argument("--repeat", type=int, default=5, help="조건별 반복 횟수")
    parser.add_argument("--start", type=int, default=30, help="각 클래스에서 몇 번째 파일부터 고를지")
    parser.add_argument("--poll", type=float, default=0.1, help="비동기 상태 조회 간격(초)")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--out", default="reports/data")
    args = parser.parse_args()

    n = args.repeat
    images = pick_images(Path(args.data), 2 + 2 * n, args.start)
    warm_sync, warm_async = images[0], images[1]
    sync_imgs, async_imgs = images[2:2 + n], images[2 + n:2 + 2 * n]

    rows, warmups = [], []
    with httpx.Client(timeout=args.timeout) as client:
        print("[워밍업] web·worker 모델 로딩 (비교에서 제외)")
        warmups.append(run_sync(client, args.base, warm_sync, 0, None))
        warmups.append(run_async(client, args.base, warm_async, 0, None, args.poll, args.timeout))
        for w in warmups:
            print(f"  {w['path']:<5} http={w['http_ms']}ms final={w['final_ms']}ms inference={w['inference_ms']}")

        for label, expect in (("MISS", "MISS"), ("HIT", "HIT")):
            print(f"[동기 {label}] {n}회")
            for i, img in enumerate(sync_imgs, 1):
                rows.append(run_sync(client, args.base, img, i, expect))
        for label, expect in (("MISS", "MISS"), ("HIT", "HIT")):
            print(f"[비동기 {label}] {n}회")
            for i, img in enumerate(async_imgs, 1):
                rows.append(run_async(client, args.base, img, i, expect, args.poll, args.timeout))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("timings.csv", rows), ("timings_warmup.csv", warmups)):
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(data)

    # 조건별 중앙값 요약
    summary = []
    for path in ("sync", "async"):
        for cs in ("MISS", "HIT"):
            g = [r for r in rows if r["path"] == path and r["cache_status"] == cs]
            summary.append({"path": path, "cache_status": cs, "n": len(g),
                            "http_ms_median": median([r["http_ms"] for r in g]),
                            "final_ms_median": median([r["final_ms"] for r in g]),
                            "inference_ms_median": median([r["inference_ms"] for r in g])})
    meta = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "base": args.base,
            "repeat": n, "poll_sec": args.poll,
            "model_id": rows[0]["model_id"] if rows else None, "summary": summary}
    (out / "timings_summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n모델: {meta['model_id']} / 반복 {n}회 / 비동기 조회 간격 {args.poll}초")
    print(f"{'경로':<6}{'캐시':<6}{'n':>3}{'HTTP 중앙값(ms)':>18}{'완료 중앙값(ms)':>18}{'추론 중앙값(ms)':>18}")
    for s in summary:
        print(f"{s['path']:<6}{s['cache_status']:<6}{s['n']:>3}{str(s['http_ms_median']):>18}"
              f"{str(s['final_ms_median']):>18}{str(s['inference_ms_median']):>18}")
    print(f"\n저장: {out/'timings.csv'}, {out/'timings_warmup.csv'}, {out/'timings_summary.json'}")


if __name__ == "__main__":
    main()
