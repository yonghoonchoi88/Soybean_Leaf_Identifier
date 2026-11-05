"""reports/data 의 실제 결과 파일만 읽어서 README·산출물용 그림을 만든다. (숫자 수기 입력 없음)

사용: python scripts/make_figures.py   (저장소 루트에서)
출력: reports/figures/pipeline.png, dashboard.png(모델), service.png(서비스), val_vs_test.png, error_types.png, latency.png,
      confusion_E1_E3.png, training_curves.png
      + 영문 README용 pipeline.en.png, dashboard.en.png, service.en.png (같은 데이터, 글자만 영어)
"""
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ROOT = Path(".")
DATA = ROOT / "reports/data"
FIG = ROOT / "reports/figures"
EVI = ROOT / "reports/evidence"

# ---------- 스타일 ----------
for name in ("Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic"):
    if any(name in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
CARD = "#f3f2ee"
EXP = {"E1": "#2a78d6", "E2": "#eb6834", "E3": "#1baf7a"}          # 실험 = 범주색 1~3번 고정
LANG = "ko"   # "en" 이면 영문 README용 그림 (set_lang 으로 전환)


def L(ko, en):
    """현재 언어에 맞는 문구를 고른다."""
    return en if LANG == "en" else ko


def set_lang(lang):
    global LANG, EXP_NAME, EXP_DESC, SHORT
    LANG = lang
    EXP_NAME = {"E1": "E1 Head-only", "E2": "E2 Full", "E3": L("E3 Full+증강", "E3 Full+Aug")}
    EXP_DESC = {"E1": L("헤드만 학습", "head only"), "E2": L("전체 층 · lr↓", "all layers · lr↓"),
                "E3": L("+ 반전·회전·밝기", "+ flip·rotate·light")}
    SHORT = {"angular_leaf_spot": L("반점병\nangular", "angular\nleaf spot"),
             "bean_rust": L("녹병\nbean_rust", "bean\nrust"), "healthy": L("정상\nhealthy", "healthy")}


def suffix():
    return ".en" if LANG == "en" else ""
STATUS = {"miss": "#d03b3b", "false_alarm": "#fab219", "confuse": "#a8a7a1"}
TAGS = {"E1": "E1_headonly", "E2": "E2_full", "E3": "E3_full_aug"}
CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]
set_lang("ko")


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def style(ax, ygrid=True):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=10, length=0)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def rounded_bars(ax, xs, hs, width, color, horizontal=False):
    """끝이 둥근 얇은 막대 (기준선 쪽은 각지게)"""
    for x, h in zip(xs, hs):
        if horizontal:
            ax.barh(x, h, height=width, color=color, edgecolor=SURFACE, linewidth=2)
        else:
            ax.bar(x, h, width=width, color=color, edgecolor=SURFACE, linewidth=2)


# ---------- 데이터 ----------
ev = {s: {e: load(f"eval_{s}_{TAGS[e]}.json") for e in EXP} for s in ("val", "test")}
tr = {e: load(f"training_report_{TAGS[e]}.json") for e in EXP}
timing_rows = list(csv.DictReader(open(DATA / "timings.csv", encoding="utf-8")))
hsv = load("hsv_rule_summary.json")


def error_types(report):
    cm = report["confusion_matrix(row=true,col=pred)"]
    miss = cm["angular_leaf_spot"]["healthy"] + cm["bean_rust"]["healthy"]
    fa = cm["healthy"]["angular_leaf_spot"] + cm["healthy"]["bean_rust"]
    conf = cm["angular_leaf_spot"]["bean_rust"] + cm["bean_rust"]["angular_leaf_spot"]
    return miss, fa, conf


def latency():
    """(key, 화면 이름, 5회 값, 중앙값) — key 는 언어와 무관하게 고정"""
    groups = {}
    for r in timing_rows:
        if r["path"] == "sync":
            groups.setdefault(f"sync_{r['cache_status']}", []).append(float(r["http_ms"]))
        else:
            groups.setdefault(f"accept_{r['cache_status']}", []).append(float(r["http_ms"]))
            groups.setdefault(f"done_{r['cache_status']}", []).append(float(r["final_ms"]))
    names = {"sync_MISS": L("동기 MISS", "Sync MISS"), "sync_HIT": L("동기 HIT", "Sync HIT"),
             "accept_MISS": L("비동기 접수 (MISS)", "Async accept (MISS)"),
             "done_MISS": L("비동기 완료 (MISS)", "Async done (MISS)")}
    return [(k, names[k], groups[k], statistics.median(groups[k])) for k in names]


# ---------- 개별 차트 ----------
def chart_val_test(ax):
    """모델 선택은 Validation 으로만, Test 는 선택된 E3 만 1회"""
    style(ax)
    xs = [0, 1, 2]
    for i, e in zip(xs, EXP):
        v = ev["val"][e]["accuracy"]
        ax.bar(i, v, 0.5, color=EXP[e], alpha=0.5, edgecolor=SURFACE, linewidth=2)
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=11, color=INK)
    t = ev["test"]["E3"]["accuracy"]
    ax.bar(3.2, t, 0.5, color=EXP["E3"], edgecolor=SURFACE, linewidth=2)
    ax.text(3.2, t + 0.012, f"{t:.3f}", ha="center", fontsize=12, color=INK, fontweight="bold")
    ax.axvline(2.6, color=GRID, linewidth=1.2, linestyle="--")
    ax.text(1.0, 1.045, L("모델 선택 · Validation", "Model selection · Validation"), ha="center", fontsize=10.5, color=INK2)
    ax.text(3.2, 1.045, L("최종 · Test 1회", "Final · Test once"), ha="center", fontsize=10.5, color=INK2)
    ax.set_xticks(xs + [3.2], [f"{EXP_NAME[e]}\n{EXP_DESC[e]}" for e in EXP]
                  + [L("E3 (선택 모델)\n처음 보는 128장", "E3 (selected)\n128 unseen images")],
                  fontsize=10.5, color=INK)
    ax.set_ylim(0.5, 1.08)
    ax.set_ylabel(L("정확도", "Accuracy"), color=INK2)


def chart_error_types(ax):
    style(ax, ygrid=False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    labels = [(L("질병 → healthy (병을 놓침)", "disease → healthy (missed)"), "miss"),
              (L("healthy → 질병 (오경보)", "healthy → disease (false alarm)"), "false_alarm"),
              (L("질병 ↔ 질병 (병명 혼동)", "disease ↔ disease (confused)"), "confuse")]
    ys = list(range(len(EXP)))[::-1]
    for y, e in zip(ys, EXP):
        vals = error_types(ev["val"][e])
        left = 0
        for (lab, key), v in zip(labels, vals):
            if v:
                ax.barh(y, v, left=left, height=0.5, color=STATUS[key], edgecolor=SURFACE, linewidth=2)
                ax.text(left + v / 2, y, str(v), ha="center", va="center", fontsize=10,
                        color="white" if key == "miss" else INK, fontweight="bold")
            left += v
        ax.text(left + 0.6, y, L(f"총 {sum(vals)}", f"total {sum(vals)}"), va="center", fontsize=11, color=INK)
    ax.set_yticks(ys, [EXP_NAME[e] for e in EXP], fontsize=11, color=INK)
    ax.set_xlim(0, 36)
    ax.set_xlabel(L("Validation 오답 수 (133장 중)", "Validation errors (out of 133)"), color=INK2)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=STATUS[k], label=l) for l, k in labels], loc="lower right",
              frameon=False, fontsize=9.5)


def chart_confusion(ax, report, title):
    cm = report["confusion_matrix(row=true,col=pred)"]
    mat = [[cm[t][p] for p in CLASSES] for t in CLASSES]
    ax.imshow(mat, cmap="Greens", vmin=0, vmax=45)
    for i in range(3):
        for j in range(3):
            v = mat[i][j]
            color = "white" if v > 25 else (STATUS["miss"] if (j == 2 and i < 2 and v) else INK)
            ax.text(j, i, str(v), ha="center", va="center", fontsize=13, color=color, fontweight="bold")
    ax.set_xticks(range(3), [SHORT[c] for c in CLASSES], fontsize=9, color=INK2)
    ax.set_yticks(range(3), [SHORT[c] for c in CLASSES], fontsize=9, color=INK2)
    ax.set_xlabel(L("예측", "Predicted"), color=INK2)
    ax.set_ylabel(L("정답", "True"), color=INK2, rotation=0, ha="right", va="center", labelpad=6)
    ax.set_title(title, fontsize=12, color=INK, pad=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def chart_latency(ax):
    style(ax, ygrid=False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    data = latency()
    colors = ["#5f5e5a", "#1baf7a", "#2a78d6", "#9a9993"]
    ys = list(range(len(data)))[::-1]
    for y, (_, name, vals, med), c in zip(ys, data, colors):
        ax.barh(y, med, height=0.5, color=c, edgecolor=SURFACE, linewidth=2)
        ax.scatter(vals, [y] * len(vals), s=14, color=INK, zorder=3, alpha=0.55)
        ax.text(max(vals) + 2, y, f"{med:.1f} ms", va="center", fontsize=11, color=INK, fontweight="bold")
    ax.set_yticks(ys, [d[1] for d in data], fontsize=10.5, color=INK)
    ax.set_xlim(0, 150)
    ax.set_xlabel(L("응답 시간 (ms) · 막대 = 중앙값, 점 = 5회 측정값",
                    "Response time (ms) · bar = median, dots = 5 runs"), color=INK2)


def chart_curves(ax):
    style(ax)
    for e in EXP:
        h = tr[e]["history"]
        xs = [r["epoch"] for r in h]
        ys = [r["val_accuracy"] for r in h]
        ax.plot(xs, ys, color=EXP[e], linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, label=EXP_NAME[e])
        best = ys.index(max(ys))   # 제공 코드가 저장한 epoch (acc > best_acc 인 첫 최고점)
        ax.annotate(f"best {ys[best]:.3f}", (xs[best], ys[best]), xytext=(6, 8 if e != "E2" else -14),
                    textcoords="offset points", fontsize=10, color=INK)
    ax.set_xticks([1, 2, 3], ["epoch 1", "epoch 2", "epoch 3"])
    ax.set_xlim(0.8, 3.45)
    ax.set_ylim(0.6, 1.02)
    ax.set_ylabel(L("Validation 정확도", "Validation accuracy"), color=INK2)
    ax.legend(frameon=False, fontsize=10, loc="lower right")


def save_single(fn, draw, size=(7.5, 4.2), title=None, note=None):
    fig, ax = plt.subplots(figsize=size, dpi=150, facecolor=SURFACE)
    draw(ax)
    if title:
        fig.suptitle(title, x=0.02, ha="left", fontsize=14, color=INK, fontweight="bold")
    if note:
        fig.text(0.02, 0.01, note, fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 0.94 if title else 1))
    fig.savefig(FIG / fn, facecolor=SURFACE)
    plt.close(fig)


# ---------- 대시보드 ----------
def card(fig, x, y, w, h, kicker, title, big, sub):
    ax = fig.add_axes([x, y, w, h])
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.06",
                                transform=ax.transAxes, facecolor=CARD, edgecolor="none"))
    ax.text(0.07, 0.80, f"{kicker} {title}", fontsize=12, color=INK2, fontweight="bold", transform=ax.transAxes)
    ax.text(0.07, 0.40, big, fontsize=31, color=INK, fontweight="bold", transform=ax.transAxes)
    ax.text(0.07, 0.13, sub, fontsize=10.5, color=INK2, transform=ax.transAxes)


def section(fig, x, y, num, title, sub):
    fig.text(x, y, f"{num}  {title}", fontsize=16, color=INK, fontweight="bold")
    fig.text(x, y - 0.014, sub, fontsize=11, color=INK2)


def image_ax(fig, rect, img, caption):
    ax = fig.add_axes(rect)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(caption, fontsize=10, color=INK2, pad=4)


def dashboard():
    """① 모델 결과 보드 — 학습·선택·평가만 (서비스 검증은 service_board)"""
    W, H = 13, 13.5
    fig = plt.figure(figsize=(W, H), dpi=150, facecolor=SURFACE)
    t3, v0, v3 = ev["test"]["E3"], ev["val"]["E1"], ev["val"]["E3"]
    fig.text(0.045, 0.958, L("콩잎 질병 분류 · 모델 결과 한눈에 보기", "Soybean Leaf Disease Classification · Model Results"),
             fontsize=26, color=INK, fontweight="bold")
    fig.text(0.045, 0.932, L("ResNet-18 Fine-tuning  ·  콩잎 1,295장(학습 1,034 / 검증 133 / 테스트 128)  ·  3클래스  ·  "
                             "Validation으로 선택 → 선택된 E3를 Test로 최종 평가",
                             "ResNet-18 fine-tuning  ·  1,295 leaf images (train 1,034 / val 133 / test 128)  ·  3 classes  ·  "
                             "chosen on Validation → selected E3 evaluated once on Test"), fontsize=12, color=INK2)

    acc = t3["accuracy"]
    f1 = sum(c["f1"] for c in t3["per_class"].values()) / 3
    miss0, miss3, miss_t = error_types(v0)[0], error_types(v3)[0], error_types(t3)[0]
    conf_t = error_types(t3)[2]
    cw, ch, cy = 0.215, 0.105, 0.79
    cards = [(L("최종 ·", "Final ·"), L("Test 정확도", "Test accuracy"), f"{acc:.3f}",
              L(f"E3 · 처음 보는 128장 · {t3['correct']}장 정답", f"E3 · 128 unseen images · {t3['correct']} correct")),
             (L("선택 ·", "Selection ·"), L("Validation 정확도", "Val accuracy"),
              f"+{(v3['accuracy'] - v0['accuracy']) * 100:.1f}%p",
              L(f"{v0['accuracy']:.3f} → {v3['accuracy']:.3f} (Head-only → Full+증강)",
                f"{v0['accuracy']:.3f} → {v3['accuracy']:.3f} (Head-only → Full+Aug)")),
             (L("안전 ·", "Safety ·"), L("병든 잎 놓침", "Missed diseases"), f"{miss0} → {miss3}",
              L(f"Validation 기준 · 최종 Test {miss_t}건", f"on Validation · final Test: {miss_t}")),
             (L("균형 ·", "Balance ·"), "Test Macro-F1", f"{f1:.3f}",
              L(f"남은 오답 {conf_t}장 모두 질병 간 혼동", f"all {conf_t} errors: disease ↔ disease"))]
    for i, c in enumerate(cards):
        card(fig, 0.045 + i * (cw + 0.0133), cy, cw, ch, *c)
    fig.add_artist(plt.Line2D([0.045, 0.955], [0.77, 0.77], color=GRID, linewidth=1))

    section(fig, 0.045, 0.745, "①", L("모델 선택 → 최종 평가", "Model selection → final evaluation"),
            L("E1→E2: Backbone까지 학습(lr 0.00005) · E2→E3: Train 증강 추가 → Val로 E3 선택",
              "E1→E2: unfreeze backbone, lr 0.00005 · E2→E3: + train augmentation"))
    chart_val_test(fig.add_axes([0.07, 0.53, 0.39, 0.165]))
    section(fig, 0.53, 0.745, "②", L("오류 유형 (Validation)", "Error types (Validation)"),
            L(f"가장 위험한 '병을 놓침'이 {miss0} → {miss3}", f"The riskiest error, a missed disease: {miss0} → {miss3}"))
    chart_error_types(fig.add_axes([0.66, 0.53, 0.29, 0.165]))

    box = fig.add_axes([0.045, 0.385, 0.91, 0.1])
    box.axis("off")
    box.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.03",
                                 transform=box.transAxes, facecolor=CARD, edgecolor="none"))
    box.text(0.02, 0.76, L("결과를 읽는 법", "How to read the results"), fontsize=14, color=INK, fontweight="bold",
             transform=box.transAxes)
    gain = v3['correct'] - ev['val']['E2']['correct']
    notes = [(L("가. Test < Validation", "a. Test < Val"), L("예상됨", "expected"),
              L(f"Val은 epoch·모델 선택에 쓰여 낙관적 · {v3['accuracy']:.3f} → {acc:.3f}",
                f"Val also chose the model → optimistic · {v3['accuracy']:.3f} → {acc:.3f}")),
             (L("나. 증강 효과", "b. Augmentation"), L("확정 안 함", "not confirmed"),
              L(f"Val +{gain}장(선택 규칙 충족) · 시드 1개 1회 실행", f"Val +{gain} images (meets the rule) · 1 seed, 1 run")),
             (L("다. 작은 표본", "c. Small sample"), L("한계", "limitation"),
              L("Val 1장 = 0.75%p · Test 1장 = 0.78%p", "1 Val image = 0.75%p · 1 Test image = 0.78%p"))]
    for i, (h, tag, body) in enumerate(notes):
        x0 = 0.02 + i * 0.33
        box.text(x0, 0.43, h, fontsize=12, color=INK, fontweight="bold", transform=box.transAxes)
        box.text(x0 + 0.165, 0.43, tag, fontsize=10,
                 color="#0ca30c" if i == 0 else MUTED, transform=box.transAxes)
        box.text(x0, 0.14, body, fontsize=10.5, color=INK2, transform=box.transAxes)

    section(fig, 0.045, 0.345, "③", L("혼동행렬", "Confusion matrices"),
            L("행=정답·열=예측 · E1: 질병 혼동+정상으로 놓침 → E3: 주로 녹병→반점병",
              "rows = true · cols = predicted · E1: confused + missed → E3: mostly rust → angular"))
    chart_confusion(fig.add_axes([0.075, 0.07, 0.19, 0.23]), v0, f"E1 · Validation {v0['correct']}/133")
    ax_r = fig.add_axes([0.285, 0.07, 0.19, 0.23])
    chart_confusion(ax_r, t3, L(f"E3 · 최종 Test {t3['correct']}/128", f"E3 · final Test {t3['correct']}/128"))
    ax_r.set_yticks([])
    for a in fig.axes[-2:]:
        a.set_xlabel("")
        a.set_ylabel("")
    section(fig, 0.53, 0.345, "④", L("학습 곡선 (Epoch별 Validation)", "Learning curves (Validation per epoch)"),
            L("3 epoch 안에서 가장 높은 epoch의 모델을 저장", "the model from the best of 3 epochs is saved"))
    chart_curves(fig.add_axes([0.59, 0.085, 0.36, 0.215]))

    fig.text(0.045, 0.018, L("출처: reports/data/ eval_*.json · training_report_E*.json  ·  "
                             "모델은 Validation으로 선택, 최종 성능은 선택된 E3의 Test 결과",
                             "Source: reports/data/ eval_*.json · training_report_E*.json  ·  "
                             "model chosen on Validation; final performance = selected E3 on Test"), fontsize=9.5, color=MUTED)
    fig.savefig(FIG / f"dashboard{suffix()}.png", facecolor=SURFACE)
    plt.close(fig)


def service_board():
    """② 서비스 검증 보드 — 실제 웹 화면·API 오류 처리·응답 시간·장애 테스트"""
    W, H = 13, 17
    fig = plt.figure(figsize=(W, H), dpi=150, facecolor=SURFACE)
    fig.text(0.045, 0.965, L("콩잎 분류 서비스 · 검증 한눈에 보기", "Soybean Leaf Classification Service · Verification"),
             fontsize=26, color=INK, fontweight="bold")
    fig.text(0.045, 0.944, L("FastAPI 동기 · Celery 비동기 · Redis 캐시/브로커 · PostgreSQL 작업 기록 · "
                             "Docker Compose 4서비스  ·  실제 실행 결과",
                             "FastAPI sync · Celery async · Redis cache/broker · PostgreSQL job records · "
                             "Docker Compose, 4 services  ·  real run results"), fontsize=12, color=INK2)

    lat = {k: m for k, _, _, m in latency()}
    cut = (lat["sync_MISS"] - lat["sync_HIT"]) / lat["sync_MISS"] * 100
    cw, ch, cy = 0.215, 0.085, 0.835
    cards = [(L("검증 ·", "Verified ·"), L("실환경 테스트", "Live tests"), "27 / 27",
              L("PASS · 모의 테스트 4 passed 별도", "PASS · plus 4 mock tests passed")),
             (L("속도 ·", "Speed ·"), L("캐시 HIT 응답", "Cache HIT response"), f"−{cut:.0f}%",
              L(f"동기 {lat['sync_MISS']:.1f} → {lat['sync_HIT']:.1f} ms (중앙값 5회)",
                f"sync {lat['sync_MISS']:.1f} → {lat['sync_HIT']:.1f} ms (median of 5)")),
             (L("응답 ·", "Response ·"), L("비동기 접수", "Async accept"), f"{lat['accept_MISS']:.1f} ms",
              L(f"job_id 먼저 반환 · 완료까지 {lat['done_MISS']:.0f} ms",
                f"returns job_id first · done in {lat['done_MISS']:.0f} ms")),
             (L("안정 ·", "Resilience ·"), L("장애 → 복구", "Outage → recovery"), "503 → done",
              L("Redis 중지 시 failed 기록 · 복구 후 정상", "Redis down → failed row · recovered"))]
    for i, c in enumerate(cards):
        card(fig, 0.045 + i * (cw + 0.0133), cy, cw, ch, *c)
    fig.add_artist(plt.Line2D([0.045, 0.955], [0.818, 0.818], color=GRID, linewidth=1))

    # ① 실제 웹 화면 (전체 폭)
    section(fig, 0.045, 0.795, "①", L("실제 웹 화면 (/vision/sync/ · /vision/async/)",
                                      "Real web pages (/vision/sync/ · /vision/async/) · UI in Korean"),
            L("같은 사진 두 번 → 두 번째는 캐시 HIT로 더 빠름 · 비동기는 job_id 접수 후 done",
              "same photo twice → the 2nd is a faster cache HIT · async: job_id accepted, then done"))
    shots = [("05_web_api/web_sync_miss.png", (35, 130, 700, 505),
              L("동기 1회차 · 37 ms · cache_hit false (MISS)", "Sync, 1st · 37 ms · cache_hit false (MISS)")),
             ("05_web_api/web_sync_hit.png", (40, 130, 705, 505),
              L("동기 2회차 · 10 ms · cache_hit true (HIT)", "Sync, 2nd · 10 ms · cache_hit true (HIT)")),
             ("05_web_api/web_async_2_done.png", (45, 130, 710, 475),
              L("비동기 · 접수 19 ms → 상태 done", "Async · accepted in 19 ms → status done"))]
    for i, (p, box_, cap) in enumerate(shots):
        img = Image.open(EVI / p).convert("RGB").crop(box_)
        image_ax(fig, [0.045 + i * 0.3067, 0.6, 0.29, 0.15], img, cap)

    fig.add_artist(plt.Line2D([0.045, 0.955], [0.56, 0.56], color=GRID, linewidth=1))
    # ② 응답 시간
    section(fig, 0.045, 0.537, "②", L("응답 시간 (E3 모델, Docker CPU)", "Response time (E3 model, Docker CPU)"),
            L("비동기는 '접수'가 빠를 뿐 결과는 더 늦다", "async is only faster to accept; the result comes later"))
    chart_latency(fig.add_axes([0.16, 0.315, 0.30, 0.18]))
    # ③ API 오류 처리
    section(fig, 0.53, 0.537, "③", L("API 입력 검증 · 오류 응답 (curl)", "API input checks · error responses (curl)"),
            L("200 정상 · 400 이미지 아님 · 413 5MB 초과 · 404 없는 작업",
              "200 OK · 400 not an image · 413 over 5 MB · 404 no such job"))
    for i, (p, cap) in enumerate([("04_sync_api_01/api_sync_errors.png", L("동기 API · 200 / 400 / 413", "Sync API · 200 / 400 / 413")),
                                  ("05_web_api/api_async_errors.png", L("비동기 API · 404 / 400 / 413", "Async API · 404 / 400 / 413"))]):
        img = Image.open(EVI / p).convert("RGB")
        image_ax(fig, [0.53, 0.405 - i * 0.107, 0.425, 0.085], img, cap)

    fig.add_artist(plt.Line2D([0.045, 0.955], [0.285, 0.285], color=GRID, linewidth=1))
    # ④ 장애·운영 테스트
    section(fig, 0.045, 0.262, "④", L("장애 · 운영 테스트", "Failure & operations tests"),
            L("일부러 멈추고 바꿔 보며 기록·복구 확인", "parts stopped or swapped on purpose to check records and recovery"))
    rows = [(L("worker 정지", "Worker stopped"),
             L("202 접수 · queued 유지 → worker 재시작 후 done (Redis 대기열 보존)",
               "202 accepted · stays queued → done after worker restart (Redis queue kept)")),
            (L("Redis 정지 · 동기", "Redis down · sync"),
             L("200 · 캐시 없이 추론 (응답 8.06초 지연 → 개선안으로 제안)",
               "200 · inference without cache (8.06 s delay → fix proposed)")),
            (L("Redis 정지 · 비동기", "Redis down · async"),
             L("503 · image_jobs에 failed와 오류 메시지 기록 (20.67초)",
               "503 · failed + error message saved in image_jobs (20.67 s)")),
            (L("Redis 복구", "Redis restored"),
             L("worker 자동 재연결 · 새 작업 done", "worker reconnects automatically · new jobs done")),
            (L("모델 교체 E1 → E3", "Model swap E1 → E3"),
             L(".env 변경 → web·worker 재생성 · 응답·DB의 model_id 모두 E3",
               ".env change → web·worker recreated · model_id is E3 in responses and DB")),
            (L("모의 테스트", "Mock tests"),
             L("pytest 4 passed · 전체 흐름 · 브로커 오류 503 · 캐시 · worker 실패 기록",
               "pytest 4 passed · full flow · broker error 503 · cache · worker failure record"))]
    for i, (h, b) in enumerate(rows):
        y = 0.218 - i * 0.031
        tag = fig.add_axes([0.045, y - 0.006, 0.05, 0.022])
        tag.axis("off")
        tag.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.25",
                                     transform=tag.transAxes, facecolor="#e3f4ea", edgecolor="none"))
        tag.text(0.5, 0.5, "PASS", ha="center", va="center", fontsize=9.5, color="#0b7a3e",
                 fontweight="bold", transform=tag.transAxes)
        fig.text(0.11, y, h, fontsize=11.5, color=INK, fontweight="bold")
        fig.text(0.29, y, b, fontsize=11, color=INK2)

    fig.text(0.045, 0.014, L("출처: reports/07_기능테스트_오류분석.md · reports/data/timings.csv · "
                             "reports/evidence/ (04_sync_api_01 · 05_web_api · 08_failure_test)  ·  웹 화면 3장은 서비스 모델 교체(E3) 전 E1로 촬영",
                             "Source: reports/07 (function tests) · reports/data/timings.csv · "
                             "reports/evidence/ (04_sync_api_01 · 05_web_api · 08_failure_test)  ·  web screenshots taken with E1, before the E3 swap"),
             fontsize=9.5, color=MUTED)
    fig.savefig(FIG / f"service{suffix()}.png", facecolor=SURFACE)
    plt.close(fig)


def pipeline():
    """README 맨 위 한 줄 파이프라인"""
    from matplotlib.patches import FancyArrowPatch
    fig = plt.figure(figsize=(13, 3.6), dpi=150, facecolor="#ffffff")
    bg = fig.add_axes([0, 0, 1, 1])
    bg.axis("off")
    bg.set_xlim(0, 1)
    bg.set_ylim(0, 1)
    bg.add_patch(FancyBboxPatch((0.01, 0.03), 0.98, 0.94, boxstyle="round,pad=0,rounding_size=0.03",
                                facecolor="#f3f2ee", edgecolor="none"))
    boxes = [(L("데이터", "Data"), L("콩잎 사진 1,295장", "1,295 soybean leaf photos"),
              L("3클래스 · 중복·누출 0", "3 classes · 0 duplicates/leaks"), "#8a8984"),
             (L("① 모델 · ResNet-18", "① Model · ResNet-18"), L("Head-only → Full → +증강", "Head-only → Full → +Aug"),
              "Val 0.767 → 0.993 · Test 0.953", EXP["E1"]),
             (L("② 서비스 · FastAPI", "② Service · FastAPI"), L("동기 · 비동기 · 캐시 · DB", "sync · async · cache · DB"),
              L("캐시 −91% · 테스트 27/27", "cache −91% · tests 27/27"), EXP["E2"]),
             (L("오답 분석", "Error analysis"), "Grad-CAM · OpenCV HSV",
              L("녹병↔반점병 혼동 확인", "rust ↔ angular confusion"), EXP["E3"])]
    bw, gap, x0, y0, bh = 0.2, 0.045, 0.04, 0.42, 0.45
    centers = []
    for i, (t, l1, l2, c) in enumerate(boxes):
        x = x0 + i * (bw + gap)
        bg.add_patch(FancyBboxPatch((x, y0), bw, bh, boxstyle="round,pad=0,rounding_size=0.015",
                                    facecolor="#ffffff", edgecolor="#d9d8d3", linewidth=1))
        bg.add_patch(FancyBboxPatch((x + 0.004, y0 + bh - 0.035), bw - 0.008, 0.03,
                                    boxstyle="round,pad=0,rounding_size=0.006", facecolor=c, edgecolor="none"))
        cx = x + bw / 2
        centers.append(cx)
        bg.text(cx, y0 + bh - 0.12, t, ha="center", fontsize=15, color=INK, fontweight="bold")
        bg.text(cx, y0 + 0.17, l1, ha="center", fontsize=11, color=INK2)
        bg.text(cx, y0 + 0.06, l2, ha="center", fontsize=11, color=c if c != "#8a8984" else INK2,
                fontweight="bold" if c != "#8a8984" else "normal")
        if i:
            bg.add_patch(FancyArrowPatch((x - gap + 0.008, y0 + bh / 2), (x - 0.008, y0 + bh / 2),
                                         arrowstyle="-|>", mutation_scale=16, color="#8a8984", linewidth=1.6))
    # 되돌아가는 점선: 오답 분석 → 모델
    green = EXP["E3"]
    ya = 0.2
    bg.plot([centers[3], centers[3]], [y0, ya], color=green, linestyle="--", linewidth=1.8)
    bg.plot([centers[3], centers[1]], [ya, ya], color=green, linestyle="--", linewidth=1.8)
    bg.add_patch(FancyArrowPatch((centers[1], ya), (centers[1], y0 - 0.01), arrowstyle="-|>",
                                 mutation_scale=16, color=green, linewidth=1.8, linestyle="--"))
    bg.text((centers[1] + centers[3]) / 2, ya - 0.08,
            L("다음 개선 방향 · 고해상도 입력 · 질병 간 혼동 데이터 보강",
              "Next · higher-resolution input · more samples of the confused diseases"),
            ha="center", fontsize=11.5, color=green, fontweight="bold")
    fig.savefig(FIG / f"pipeline{suffix()}.png", facecolor="#ffffff")
    plt.close(fig)

if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    save_single("val_vs_test.png", chart_val_test, title="모델 선택(Validation) → 최종 평가(Test 1회)",
                note="E1·E2·E3는 Validation으로 비교·선택, 최종 평가는 선택된 E3")
    save_single("error_types.png", chart_error_types, title="Validation 오류 유형 (133장)")
    save_single("latency.png", chart_latency, title="응답 시간 비교 (E3 서비스 · Docker CPU · 조건별 5회)",
                note="비동기 완료 시간은 상태 조회 간격 0.1초 기준 (브라우저 화면은 1초 간격)")
    save_single("training_curves.png", chart_curves, title="Epoch별 Validation 정확도")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3), dpi=150, facecolor=SURFACE)
    chart_confusion(axes[0], ev["val"]["E1"], f"E1 · Validation {ev['val']['E1']['correct']}/133")
    chart_confusion(axes[1], ev["val"]["E3"], f"E3 · Validation {ev['val']['E3']['correct']}/133")
    chart_confusion(axes[2], ev["test"]["E3"], f"E3 · 최종 Test {ev['test']['E3']['correct']}/128")
    fig.suptitle("혼동행렬 (빨간 숫자 = 병든 잎을 healthy로 놓침)", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "confusion_E1_E3.png", facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    dashboard()
    service_board()
    pipeline()
    # 영문 README(README.en.md)용: 같은 데이터로 글자만 영어인 보드 3장
    set_lang("en")
    dashboard()
    service_board()
    pipeline()
    set_lang("ko")
    print("saved:", sorted(p.name for p in FIG.glob("*.png")))
