"""reports/data 의 실제 결과 파일만 읽어서 README·산출물용 그림을 만든다. (숫자 수기 입력 없음)

사용: python scripts/make_figures.py   (저장소 루트에서)
출력: reports/figures/dashboard.png, val_vs_test.png, error_types.png, latency.png,
      confusion_E0_E3.png, training_curves.png
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
EXP = {"E0": "#2a78d6", "E2": "#eb6834", "E3": "#1baf7a"}          # 실험 = 범주색 1~3번 고정
EXP_NAME = {"E0": "E0 Head-only", "E2": "E2 Full", "E3": "E3 Full+증강"}
STATUS = {"miss": "#d03b3b", "false_alarm": "#fab219", "confuse": "#a8a7a1"}
TAGS = {"E0": "E0_headonly", "E2": "E2_full", "E3": "E3_full_aug"}
CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]
SHORT = {"angular_leaf_spot": "반점병\nangular", "bean_rust": "녹병\nbean_rust", "healthy": "정상\nhealthy"}


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
    groups = {}
    for r in timing_rows:
        if r["path"] == "sync":
            groups.setdefault(f"동기 {r['cache_status']}", []).append(float(r["http_ms"]))
        else:
            groups.setdefault(f"비동기 접수 ({r['cache_status']})", []).append(float(r["http_ms"]))
            groups.setdefault(f"비동기 완료 ({r['cache_status']})", []).append(float(r["final_ms"]))
    order = ["동기 MISS", "동기 HIT", "비동기 접수 (MISS)", "비동기 완료 (MISS)"]
    return [(k, groups[k], statistics.median(groups[k])) for k in order]


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
    ax.text(1.0, 1.045, "모델 선택 · Validation", ha="center", fontsize=10.5, color=INK2)
    ax.text(3.2, 1.045, "최종 · Test 1회", ha="center", fontsize=10.5, color=INK2)
    ax.set_xticks(xs + [3.2], [EXP_NAME[e] for e in EXP] + ["E3 (선택 모델)"], fontsize=10.5, color=INK)
    ax.set_ylim(0.5, 1.08)
    ax.set_ylabel("정확도", color=INK2)


def chart_error_types(ax):
    style(ax, ygrid=False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    labels = [("질병 → healthy (병을 놓침)", "miss"), ("healthy → 질병 (오경보)", "false_alarm"),
              ("질병 ↔ 질병 (병명 혼동)", "confuse")]
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
        ax.text(left + 0.6, y, f"총 {sum(vals)}", va="center", fontsize=11, color=INK)
    ax.set_yticks(ys, [EXP_NAME[e] for e in EXP], fontsize=11, color=INK)
    ax.set_xlim(0, 36)
    ax.set_xlabel("Validation 오답 수 (133장 중)", color=INK2)
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
    ax.set_xlabel("예측", color=INK2)
    ax.set_ylabel("정답", color=INK2)
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
    for y, (name, vals, med), c in zip(ys, data, colors):
        ax.barh(y, med, height=0.5, color=c, edgecolor=SURFACE, linewidth=2)
        ax.scatter(vals, [y] * len(vals), s=14, color=INK, zorder=3, alpha=0.55)
        ax.text(max(vals) + 2, y, f"{med:.1f} ms", va="center", fontsize=11, color=INK, fontweight="bold")
    ax.set_yticks(ys, [d[0] for d in data], fontsize=10.5, color=INK)
    ax.set_xlim(0, 150)
    ax.set_xlabel("응답 시간 (ms) · 막대 = 중앙값, 점 = 5회 측정값", color=INK2)


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
    ax.set_ylabel("Validation 정확도", color=INK2)
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
    W, H = 13, 19.5
    fig = plt.figure(figsize=(W, H), dpi=150, facecolor=SURFACE)
    t3, v0, v3 = ev["test"]["E3"], ev["val"]["E0"], ev["val"]["E3"]
    fig.text(0.045, 0.972, "콩잎 질병 분류 · 프로젝트 한눈에 보기", fontsize=26, color=INK, fontweight="bold")
    fig.text(0.045, 0.956, f"ResNet-18 Fine-tuning  ·  콩잎 이미지 1,295장(학습 1,034 / 검증 133 / 테스트 128)  ·  "
             "3클래스  ·  FastAPI · Redis · Celery · PostgreSQL · Docker Compose",
             fontsize=12, color=INK2)

    # 실물 헤더: 클래스별 실제 사진(E3 예측) + 실제 서비스 화면
    feats = {r["file"]: r for r in csv.DictReader(open(DATA / "hsv_features_test.csv", encoding="utf-8"))}
    heads = []
    for c, ko in (("angular_leaf_spot", "반점병"), ("bean_rust", "녹병"), ("healthy", "정상")):
        r = feats[f"{c}_test.0.jpg"]
        heads.append((f"{c}_test.0", f"{ko} → E3 {float(r['top1_score']):.2f} {'정답' if r['correct'] == 'True' else '오답'}"))
    for i, (stem, cap) in enumerate(heads):
        p = Image.open(FIG / "hsv" / f"hsv_test_{stem}.png")
        image_ax(fig, [0.045 + i * 0.158, 0.80, 0.148, 0.135], p.crop((0, 0, 256, 256)), cap)
    shot = Image.open(EVI / "05_web_api" / "web_async_2_done.png")
    shot = shot.crop((0, 0, shot.width, int(shot.width * 0.62)))
    image_ax(fig, [0.53, 0.80, 0.425, 0.135], shot, "실제 서비스 화면 · 비동기 접수 → Celery 처리 → done")

    # 핵심 숫자 4개
    acc = t3["accuracy"]
    miss0, miss3, miss_t = error_types(v0)[0], error_types(v3)[0], error_types(t3)[0]
    lat = {k: m for k, _, m in latency()}
    cut = (lat["동기 MISS"] - lat["동기 HIT"]) / lat["동기 MISS"] * 100
    r1v, r2v = hsv["rules_val(design)"]["R1"], hsv["rules_val(design)"]["R2"]
    cw, ch, cy = 0.215, 0.072, 0.705
    cards = [("최종 ·", "Test 정확도", f"{acc:.3f}", f"E3 · 모델 확정 후 1회 · 처음 보는 128장"),
             ("선택 ·", "Validation 정확도", f"+{(v3['accuracy'] - v0['accuracy']) * 100:.1f}%p", f"{v0['accuracy']:.3f} → {v3['accuracy']:.3f} (Head-only → Full+증강)"),
             ("안전 ·", "병든 잎 놓침", f"{miss0} → {miss3}", f"Validation 기준 · 최종 Test {miss_t}건"),
             ("속도 ·", "캐시 HIT 응답", f"−{cut:.0f}%", f"동기 {lat['동기 MISS']:.1f} → {lat['동기 HIT']:.1f} ms (중앙값 5회)")]
    for i, c in enumerate(cards):
        card(fig, 0.045 + i * (cw + 0.0133), cy, cw, ch, *c)
    fig.add_artist(plt.Line2D([0.045, 0.955], [0.69, 0.69], color=GRID, linewidth=1))

    # ① 성능 변화 / ② 오류 유형
    section(fig, 0.045, 0.672, "①", "모델 선택 → 최종 평가", "Validation으로 E0·E2·E3를 비교해 E3 선택 → Test는 E3에만 1회")
    chart_val_test(fig.add_axes([0.07, 0.505, 0.39, 0.135]))
    section(fig, 0.53, 0.672, "②", "오류 유형 (Validation)", f"가장 위험한 '병을 놓침'이 {miss0} → {miss3}")
    chart_error_types(fig.add_axes([0.66, 0.505, 0.29, 0.135]))

    # 정직한 해석 박스
    box = fig.add_axes([0.045, 0.40, 0.91, 0.075])
    box.axis("off")
    box.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.03",
                                 transform=box.transAxes, facecolor=CARD, edgecolor="none"))
    box.text(0.02, 0.78, "결과를 읽는 법", fontsize=14, color=INK, fontweight="bold", transform=box.transAxes)
    notes = [("① Test < Validation", "예상됨",
              f"Val은 epoch·모델 선택에 쓰여 낙관적 · {v3['accuracy']:.3f} → {acc:.3f}"),
             ("② 증강 효과", "확정 안 함", f"Val +{v3['correct'] - ev['val']['E2']['correct']}장(선택 규칙 충족) · 시드 1개 1회 실행"),
             ("③ 작은 표본", "한계", "Val 1장 = 0.75%p · Test 1장 = 0.78%p")]
    for i, (h, tag, body) in enumerate(notes):
        x0 = 0.02 + i * 0.33
        box.text(x0, 0.45, h, fontsize=12, color=INK, fontweight="bold", transform=box.transAxes)
        box.text(x0 + 0.165, 0.45, tag, fontsize=10,
                 color="#0ca30c" if tag == "예상됨" else MUTED, transform=box.transAxes)
        box.text(x0, 0.15, body, fontsize=10.5, color=INK2, transform=box.transAxes)

    # ③ 혼동행렬 / ④ 응답시간
    section(fig, 0.045, 0.375, "③", "혼동행렬", "E0(검증)는 질병끼리 혼동 + healthy 쏠림 → E3(최종 테스트)는 녹병→반점병만 남음")
    chart_confusion(fig.add_axes([0.075, 0.21, 0.19, 0.125]), v0, f"E0 · Validation {v0['correct']}/133")
    ax_r = fig.add_axes([0.285, 0.21, 0.19, 0.125])
    chart_confusion(ax_r, t3, f"E3 · 최종 Test {t3['correct']}/128")
    ax_r.set_yticks([])
    ax_r.set_ylabel("")
    section(fig, 0.53, 0.375, "④", "응답 시간 (E3 서비스, Docker CPU)", "비동기는 '접수'가 빠를 뿐 결과는 더 늦다")
    chart_latency(fig.add_axes([0.66, 0.225, 0.29, 0.125]))

    fig.add_artist(plt.Line2D([0.045, 0.955], [0.19, 0.19], color=GRID, linewidth=1))
    # ⑤ 오류 분석
    hv = hsv["class_stats_val"]
    section(fig, 0.045, 0.172, "⑤", "남은 오답 분석 (Grad-CAM · OpenCV HSV)",
            "E3는 잎을 보고도 녹병을 반점병으로 혼동 · 색 면적비만으로는 녹병과 정상을 구분하지 못함")
    gc = Image.open(FIG / "gradcam" / "gradcam_test-wrong_bean_rust_test.28.png")
    tiles = [gc.crop((0, 0, 224, 224)), gc.crop((690, 0, 914, 224)), gc.crop((230, 0, 454, 224))]
    caps = ["원본 · 정답 녹병", "E0 근거 → healthy", "E3 근거 → 반점병"]
    for i, (t, c) in enumerate(zip(tiles, caps)):
        image_ax(fig, [0.045 + i * 0.118, 0.035, 0.11, 0.11], t, c)
    tx = 0.42
    lines = [("Grad-CAM", "E3 오답 7장: E3는 잎 위, 같은 사진에서 E0는 3장이 흙·배경 쪽 (오답만 관찰)"),
             ("HSV 병반 면적비", f"Val 중앙값  반점병 {hv['angular_leaf_spot']['lesion_ratio_median']:.4f}  ·  "
                                f"녹병 {hv['bean_rust']['lesion_ratio_median']:.4f}  ·  정상 {hv['healthy']['lesion_ratio_median']:.4f}"),
             ("R1  healthy 의심", f"Val 보류 {r1v['flagged']}장 · 오답 {r1v['errors_caught']}/{r1v['errors_total']} 포착 → 기각"),
             ("R2  점수 차 < 0.4", f"Val 보류 {r2v['flagged']}장 · 오답 {r2v['errors_caught']}/{r2v['errors_total']} 포착 → 근거 부족, 적용 보류")]
    for i, (h, b) in enumerate(lines):
        y = 0.135 - i * 0.027
        fig.text(tx, y, h, fontsize=11.5, color=INK, fontweight="bold")
        fig.text(tx + 0.13, y, b, fontsize=10.5, color=INK2)
    fig.text(0.045, 0.012, "출처: reports/data/ eval_*.json · training_report_E*.json · timings.csv · hsv_rule_summary.json  ·  "
             "모델·규칙은 Validation으로만 선택, Test는 선택된 E3 성능 보고에만 사용", fontsize=9.5, color=MUTED)
    fig.savefig(FIG / "dashboard.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    save_single("val_vs_test.png", chart_val_test, title="모델 선택(Validation) → 최종 평가(Test 1회)",
                note="E0·E2·E3는 Validation으로만 비교·선택, Test는 선택된 E3에 1회 적용")
    save_single("error_types.png", chart_error_types, title="Validation 오류 유형 (133장)")
    save_single("latency.png", chart_latency, title="응답 시간 비교 (E3 서비스 · Docker CPU · 조건별 5회)",
                note="비동기 완료 시간은 상태 조회 간격 0.1초 기준 (브라우저 화면은 1초 간격)")
    save_single("training_curves.png", chart_curves, title="Epoch별 Validation 정확도")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3), dpi=150, facecolor=SURFACE)
    chart_confusion(axes[0], ev["val"]["E0"], f"E0 · Validation {ev['val']['E0']['correct']}/133")
    chart_confusion(axes[1], ev["val"]["E3"], f"E3 · Validation {ev['val']['E3']['correct']}/133")
    chart_confusion(axes[2], ev["test"]["E3"], f"E3 · 최종 Test {ev['test']['E3']['correct']}/128")
    fig.suptitle("혼동행렬 (빨간 숫자 = 병든 잎을 healthy로 놓침)", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "confusion_E0_E3.png", facecolor=SURFACE)
    plt.close(fig)
    dashboard()
    print("saved:", sorted(p.name for p in FIG.glob("*.png")))
