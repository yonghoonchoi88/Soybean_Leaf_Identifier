# check_dataset.py — 클래스별 개수 + 해시 기반 중복·누출 검사 (산출물 1 증빙용)
import hashlib, json
from collections import defaultdict
from pathlib import Path

ROOT = Path("dataset")
SPLITS = ["train", "val", "test"]
EXTS = {".jpg", ".jpeg", ".png", ".webp"}   # finetune.py와 같은 확장자 기준

counts = {}                      # {split: {class: 장수}}
seen = defaultdict(list)         # {해시: [(split, class, 파일명), ...]}

for split in SPLITS:
    counts[split] = {}
    for cls_dir in sorted(p for p in (ROOT / split).iterdir() if p.is_dir()):
        files = [f for f in cls_dir.iterdir() if f.suffix.lower() in EXTS]
        counts[split][cls_dir.name] = len(files)
        for f in files:
            digest = hashlib.sha256(f.read_bytes()).hexdigest()   # 사진의 "지문"
            seen[digest].append((split, cls_dir.name, f.name))

dups = {h: locs for h, locs in seen.items() if len(locs) > 1}                          # 같은 지문 2개 이상 = 중복
leaks = {h: locs for h, locs in dups.items() if len({s for s, _, _ in locs}) > 1}       # 다른 split에 걸침 = 누출
label_conflicts = {h: locs for h, locs in dups.items() if len({c for _, c, _ in locs}) > 1}  # 라벨 충돌

report = {
    "counts": counts,
    "totals": {s: sum(c.values()) for s, c in counts.items()},
    "duplicate_groups": len(dups),
    "cross_split_leaks": len(leaks),
    "label_conflicts": len(label_conflicts),
    "leak_examples": list(leaks.values())[:10],
}
Path("reports/data").mkdir(parents=True, exist_ok=True)
Path("reports/data/dataset_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False, indent=2))