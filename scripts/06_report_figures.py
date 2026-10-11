"""報告用五張圖與表 1：一律由 outputs/*/tables 的 CSV 重繪（不裁切舊圖），統一字型、配色與印刷尺寸。

  圖1 國七上床時間分組 → 國八新發生憂鬱或自傷意念風險的實際發生率（prediction/tables/observed_rate_by_bin）
      半欄寬（8.4 cm），置於第 1 節。
  圖2 國七的六種晚上：LCA 六類型 18:00–02:00 逐時主要活動組成（typology/tables/type_hourly_composition、type_sizes）
      螢幕活動加白色斜線，灰階列印仍可區分「螢幕／非螢幕」。
  圖3 兩波交叉延宕模型：網路成癮、上床時間、憂鬱（mediation/tables/clpm_paths）；半欄寬（8.4 cm）。
  圖4 (a) 各類型 vs 早睡規律型的國八憂鬱差距：M2 vs M2＋國七上床時間。w2_type_vs_bedtime.csv 只存兩型點估計、
          沒有 CI，故依 02_evening_typology.py 4b 節同一設定（同一樣本、WLS/HOUWGT、學校叢集 SE）重算五型，
          並與該表核對（差異 >1e-4 即中止）；結果另存 report/tables/fig3a_type_vs_bedtime.csv（檔名沿用舊版）。
      (b) 23 點後醒著（螢幕為主／非螢幕）vs 已入睡 → 國八上床時間延後（mediation/tables/late_night_specificity）
  圖5 數位教養與一般親子 → 國八四項結果的標準化 β（parenting/tables/coef_main，主模型＝國八同期）
  表1 新發生風險預警模型表現（prediction/tables/model_performance；字串取自 prediction/results.md 以免二次四捨五入）

執行：.venv/Scripts/python -I scripts/06_report_figures.py
輸出：report/figures/fig1–fig5*.png（全欄寬 17 cm 或半欄寬 8.4 cm、300 dpi）、report/tables/table1.html、
      report/tables/fig3a_type_vs_bedtime.csv
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tigps import data as D  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = ROOT / "outputs"
FIG, TAB = ROOT / "report" / "figures", ROOT / "report" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
CM, W, WH = 1 / 2.54, 17.0, 8.4  # 全欄寬 17 cm；半欄寬 8.4 cm

INK, SEC, MUTED, GRID, BASE = "#1f1f1f", "#52514e", "#898781", "#e6e4de", "#c3c2b7"
BLUE, RED = "#2a78d6", "#d9432f"     # 主色（基本模型／顯著路徑）、強調色（關鍵路徑）
NS_GREY, LIGHT = "#a8a69f", "#bdbab3"
# 圖2 狀態配色（已以 dataviz validate_palette.py 檢查相鄰色：補習–讀書、影音–遊戲–社群 皆 PASS）
NAVY, LIFE = "#23255a", "#dcd9d1"    # NAVY＝「睡覺」色；圖4 的「再控制睡眠作息」也用它（灰階亦與 BLUE 可分）
GREEN, AMBER, PURPLE = "#1baf7a", "#eda100", "#9b3a8f"
MINUS = "–"                          # 刻度負號用 en dash（Microsoft JhengHei 無 U+2212）
plt.rcParams.update({
    "font.family": ["Microsoft JhengHei", "sans-serif"], "axes.unicode_minus": False, "font.size": 7,
    "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.5, "axes.edgecolor": BASE, "axes.labelcolor": SEC, "xtick.color": SEC, "ytick.color": SEC,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.dpi": 300, "legend.frameon": False,
    "hatch.linewidth": 0.5, "hatch.color": "white"})


def rd(p):
    return pd.read_csv(OUT / p, encoding="utf-8-sig")


def new_fig(h, w=W):
    return plt.figure(figsize=(w * CM, h * CM))


def add_ax(fig, x, y, w, h):
    """以 cm 指定座標軸位置（左下角 x, y；寬 w、高 h）。"""
    fw, fh = fig.get_size_inches() / CM
    return fig.add_axes([x / fw, y / fh, w / fw, h / fh])


def ftext(fig, x, y, s, **kw):
    fw, fh = fig.get_size_inches() / CM
    return fig.text(x / fw, y / fh, s, **kw)


def save(fig, name):
    fig.savefig(FIG / name, dpi=300, facecolor="white")
    plt.close(fig)
    print("寫出", FIG / name)


def pfmt(p):
    return "p < .001" if p < .001 else f"p = {p:.3f}".replace("0.", ".")


def tick(v, fmt):
    """刻度文字：負號改為 en dash。"""
    s = fmt.format(abs(v))
    return (MINUS + s) if v < 0 else s


# ====================================================================== 圖1（半欄）
def fig_hook():
    ob = rd("prediction/tables/observed_rate_by_bin.csv")
    ob = ob[ob["特徵"] == "bedtime"].reset_index(drop=True)
    fig = new_fig(5.25, WH)
    ax = add_ax(fig, 0.82, 1.42, WH - 0.92, 3.3)
    x = np.arange(len(ob))
    # 單一序列：一種顏色；午夜以參考線標示（不以顏色暗示門檻）
    ax.bar(x, ob["rate"], width=0.62, color=BLUE, zorder=2)
    ax.errorbar(x, ob["rate"], yerr=[ob["rate"] - ob["lo"], ob["hi"] - ob["rate"]], fmt="none", ecolor=INK,
                elinewidth=0.6, capsize=1.6, capthick=0.6, zorder=3)
    for i in (0, len(ob) - 1):
        ax.text(i, ob.loc[i, "hi"] + 0.012, f"{ob.loc[i, 'rate']:.1%}", ha="center", va="bottom", fontsize=7,
                fontweight="bold")
    ax.axvline(4.5, color=MUTED, lw=0.6, ls=(0, (2, 1.5)), zorder=1)
    ax.text(4.42, 0.497, "午夜", color=SEC, fontsize=7, va="top", ha="right")
    labs = ["22:00\n前", "22:00\n–29", "22:30\n–59", "23:00\n–29", "23:30\n–59", "0:00\n–29", "0:30\n以後"]
    ax.set_xticks(x, [f"{lab}\nn={n:,}" for lab, n in zip(labs, ob["n"])], linespacing=1.0)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(-0.55, len(ob) - 0.45)
    ax.set_ylim(0, 0.5)
    ax.set_yticks(np.arange(0, 0.51, 0.1), [f"{v:.0%}" for v in np.arange(0, 0.51, 0.1)])
    ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ftext(fig, 0.1, 5.0, "國八新發生憂鬱或自傷意念風險（加權）", fontsize=7, color=SEC, va="center")
    ftext(fig, 0.82 + (WH - 0.92) / 2, 0.16, "國七上課日上床時間", fontsize=7, color=SEC, ha="center", va="center")
    save(fig, "fig1_late_sleep_risk.png")


# ====================================================================== 圖2
def fig_types():
    comp = pd.read_csv(OUT / "typology/tables/type_hourly_composition.csv", index_col=[0, 1], encoding="utf-8-sig")
    sizes = rd("typology/tables/type_sizes.csv").set_index("type")
    order = list(sizes.index)
    stack = [("補習課輔", "補習／課輔", GREEN), ("寫作業讀書", "寫作業／讀書", BLUE), ("生活與其他", "吃飯盥洗等生活", LIFE),
             ("螢幕：影音", "螢幕：影音", AMBER), ("螢幕：遊戲電視", "螢幕：遊戲／電視", RED),
             ("螢幕：社群聊天", "螢幕：社群聊天", PURPLE), ("睡覺", "睡覺", NAVY)]
    hatch = {k: ("////" if k.startswith("螢幕") else None) for k, _, _ in stack}
    hrs = list(comp.columns)  # 18:00 … 1:00（每格為該小時起點）
    fig = new_fig(5.4)
    left, right, gap, y0, ph = 0.95, 0.12, 0.2, 1.36, 2.74
    pw = (W - left - right - gap * 5) / 6
    x = np.arange(len(hrs))
    for i, t in enumerate(order):
        ax = add_ax(fig, left + i * (pw + gap), y0, pw, ph)
        bottom = np.zeros(len(hrs))
        for key, _, col in stack:
            v = comp.loc[(t, key)].to_numpy(float)
            ax.bar(x, v, bottom=bottom, width=0.88, color=col, edgecolor="white", linewidth=0.35, hatch=hatch[key])
            bottom += v
        ax.set_xlim(-0.55, len(hrs) - 0.45)
        ax.set_ylim(0, 1.0)
        ax.set_xticks([0, 2, 4, 6], ["18", "20", "22", "0"])
        ax.set_yticks([0, .5, 1], ["0", "50%", "100%"] if i == 0 else [])
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="x", length=0, pad=2)
        ax.tick_params(axis="y", length=0 if i else 2)
        r = sizes.loc[t]
        aw = "<1%" if r.awake_midnight_pct < 1 else f"{r.awake_midnight_pct:.0f}%"
        name = t + ("†" if t == "作息紊亂型" else "")
        for k, (s, kw) in enumerate(((f"午夜仍醒 {aw}", dict(color=SEC)), (f"占 {r.pct_w:.1f}%", dict(color=SEC)),
                                     (name, dict(fontsize=7.5, fontweight="bold")))):
            ax.text(0, 1.0 + (0.08 + 0.3 * k) / ph, s, transform=ax.transAxes, va="bottom", **{"fontsize": 7, **kw})
    ftext(fig, left + (W - left - right) / 2, 0.86, "時段起點（時；0 ＝ 午夜 0:00–1:00）；斜線＝螢幕活動", ha="center",
          va="center", color=SEC, fontsize=7)
    handles = [Patch(fc=col, ec="white", lw=0, hatch=hatch[key], label=lab) for key, lab, col in stack[::-1]]
    fw, fh = fig.get_size_inches() / CM
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.36 / fh), ncol=7, handlelength=1.3,
               handleheight=0.9, handletextpad=0.35, columnspacing=0.9, borderaxespad=0)
    save(fig, "fig2_evening_types.png")


# ====================================================================== 圖3（半欄）
def fig_clpm():
    cl = rd("mediation/tables/clpm_paths.csv")
    cl = cl[cl["版本"] == "上床時間"].set_index(["自", "至"])
    H = 5.55
    fig = new_fig(H, WH)
    bx = add_ax(fig, 0, 0, WH, H)
    bx.set_xlim(0, WH)
    bx.set_ylim(0, H)
    bx.axis("off")
    VARS = [("addict", "網路成癮"), ("bedtime", "上床時間"), ("cesd_nosleep", "憂鬱症狀")]
    ys = {"addict": 4.12, "bedtime": 2.72, "cesd_nosleep": 1.32}
    xl, xr, bw, bh = 0.88, WH - 0.88, 1.56, 0.56     # 左右方塊中心 x、寬、高
    for xc, wave in ((xl, "國七（2023）"), (xr, "國八（2024）")):
        bx.text(xc, 4.95, wave, ha="center", va="center", fontsize=7.5, fontweight="bold")
        for v, lab in VARS:
            bx.add_patch(FancyBboxPatch((xc - bw / 2, ys[v] - bh / 2), bw, bh, boxstyle="round,pad=0,rounding_size=0.12",
                                        fc="#f4f3ef", ec=BASE, lw=0.6, zorder=3))
            bx.text(xc, ys[v], lab, ha="center", va="center", fontsize=7.5, zorder=4)
    xa, xb, off = xl + bw / 2 + 0.02, xr - bw / 2 - 0.06, 0.15

    def arrow(p0, p1, color, lw, ls="-", z=2):
        bx.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=6.5, color=color, lw=lw,
                                     linestyle=ls, shrinkA=0, shrinkB=0, zorder=z, joinstyle="miter"))

    def label(x, y, s, color=INK, bold=False):
        bx.text(x, y, s, ha="center", va="center", fontsize=7, color=color, fontweight="bold" if bold else "normal",
                zorder=5, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))

    segs, rects = {}, [(xc, ys[v], bw + 0.1, bh + 0.1) for xc in (xl, xr) for v, _ in VARS]  # 線段、需避開的矩形
    for v, _ in VARS:  # 自我迴歸
        segs[(v, v)] = ((xa, ys[v]), (xb, ys[v]))
        arrow(*segs[(v, v)], "#d6d4cd", 1.4, z=1)
    CROSS = [("bedtime", "cesd_nosleep"), ("cesd_nosleep", "bedtime"), ("addict", "cesd_nosleep"),
             ("cesd_nosleep", "addict"), ("addict", "bedtime"), ("bedtime", "addict")]   # 先放重點路徑的標籤
    for s, t in CROSS:
        down = ys[t] < ys[s]
        segs[(s, t)] = ((xa, ys[s] + (-off if down else off)), (xb, ys[t] + (off if down else -off)))

    def clearance(cx, cy, w, h, own):
        """標籤矩形與其他線段、方塊、已放標籤的最小距離（cm）。"""
        best = np.inf
        for k, (p0, p1) in segs.items():
            if k == own:
                continue
            tt = np.linspace(0, 1, 160)
            px, py = p0[0] + tt * (p1[0] - p0[0]), p0[1] + tt * (p1[1] - p0[1])
            best = min(best, np.hypot(np.maximum(abs(px - cx) - w / 2, 0), np.maximum(abs(py - cy) - h / 2, 0)).min())
        for rx, ry, rw, rh in rects:
            best = min(best, np.hypot(max(abs(rx - cx) - (w + rw) / 2, 0), max(abs(ry - cy) - (h + rh) / 2, 0)))
        return best

    for v, _ in VARS:   # 自我迴歸標籤固定放在左段
        label(xa + 0.5, ys[v], f"{cl.loc[(f'{v}_1', f'{v}_2'), 'beta']:.2f}", color=SEC)
        rects.append((xa + 0.5, ys[v], 0.5, 0.3))
    for s, t in CROSS:
        r = cl.loc[(f"{s}_1", f"{t}_2")]
        p0, p1 = segs[(s, t)]
        key, rev, sig = (s, t) == ("bedtime", "cesd_nosleep"), (s, t) == ("cesd_nosleep", "bedtime"), r.q < .05
        col = RED if key else (BLUE if sig else NS_GREY)
        arrow(p0, p1, col, 1.5 if key else (1.0 if sig else 0.8), "-" if sig else (0, (3, 2)))
        cand = [(clearance(p0[0] + u * (p1[0] - p0[0]), p0[1] + u * (p1[1] - p0[1]), 0.68, 0.3, (s, t)), u)
                for u in np.linspace(0.14, 0.86, 37)]
        u = max(cand, key=lambda c: (round(c[0], 2), -abs(c[1] - 0.5)))[1]   # 最空曠處；同分取近中點
        lx, ly = p0[0] + u * (p1[0] - p0[0]), p0[1] + u * (p1[1] - p0[1])
        label(lx, ly, f"{r.beta:.3f}", color=col if sig else SEC, bold=key or rev)
        rects.append((lx, ly, 0.68, 0.3))
    hs = [Line2D([], [], color=RED, lw=1.5, label="上床→憂鬱"), Line2D([], [], color=BLUE, lw=1.0, label="其他 q < .05"),
          Line2D([], [], color=NS_GREY, lw=0.8, ls=(0, (3, 2)), label="未達顯著"),
          Line2D([], [], color="#d6d4cd", lw=1.4, label="自我迴歸")]
    bx.legend(handles=hs, loc="center", bbox_to_anchor=(0.5, 0.33 / H), ncol=4, handlelength=1.5,
              handletextpad=0.35, columnspacing=0.8, borderaxespad=0)
    save(fig, "fig3_clpm.png")


# ====================================================================== 圖4
def refit_type_vs_bedtime():
    """重現 02_evening_typology.py 4b(1)：國八憂鬱（不含睡眠題）~ 類型 + 國七憂鬱 + 共變項（M2），再加國七上床時間。"""
    order = list(rd("typology/tables/type_sizes.csv")["type"])
    ta = rd("typology/tables/type_assignment.csv")
    P = D.load_panel().merge(ta[["nstudent_id", "type", "randA"]], on="nstudent_id", how="inner")
    P["female"] = P["female_1"].fillna(P["female_2"])
    P["tc"] = pd.Categorical(P["type"], categories=order).codes
    y, base, cov = "cesd_nosleep_2", "cesd_nosleep_1", "female + fam_econ_1 + C(digi_level)"
    d = P[P["randA"]][[y, "tc", base, "bedtime_1", "female", "fam_econ_1", "digi_level", "HOUWGT",
                       "nschool_id"]].dropna()
    fits = {spec: smf.wls(f"{y} ~ C(tc){rhs} + {base} + {cov}", d, weights=d["HOUWGT"]).fit(
        cov_type="cluster", cov_kwds={"groups": d["nschool_id"]}) for spec, rhs in (("M2", ""), ("M2+bed", " + bedtime_1"))}
    rows = []
    for spec, f in fits.items():
        ci = f.conf_int()
        for c, t in enumerate(order[1:], 1):
            p = f"C(tc)[T.{c}]"
            rows.append(dict(spec=spec, type=t, n=int(f.nobs), est=f.params[p], lo=ci.loc[p, 0], hi=ci.loc[p, 1],
                             p=f.pvalues[p]))
    res = pd.DataFrame(rows)
    chk = rd("typology/tables/w2_type_vs_bedtime.csv").set_index("outcome").loc[y]
    for t in ("電玩遊戲型", "影音社群夜貓型"):
        for spec, col in (("M2", f"{t}_M2"), ("M2+bed", f"{t}_M2+bed")):
            got = res[(res.spec == spec) & (res.type == t)]["est"].iloc[0]
            assert abs(got - chk[col]) < 1e-4, (t, spec, got, chk[col])
    assert res["n"].iloc[0] == chk["n"]
    res.round(5).to_csv(TAB / "fig3a_type_vs_bedtime.csv", index=False, encoding="utf-8-sig")
    print(f"圖4a 重算 n={res['n'].iloc[0]}，與 w2_type_vs_bedtime.csv 一致")
    return res, chk


def dot_ci(ax, y, est, lo, hi, color, marker, filled=True, ms=4.2):
    ax.plot([lo, hi], [y, y], color=color, lw=1.0, solid_capstyle="butt", zorder=2)
    ax.plot(est, y, marker=marker, ms=ms, mfc=color if filled else "white", mec=color, mew=0.9, zorder=3, ls="none")


def fig_twist():
    res, chk = refit_type_vs_bedtime()
    H = 5.0
    fig = new_fig(H)
    SPECS = (("M2", BLUE, "o", 0.16), ("M2+bed", NAVY, "D", -0.16))
    # (a)
    types = list(res["type"].unique())
    ax = add_ax(fig, 2.2, 0.95, 5.6, 2.78)
    ys = np.arange(len(types))[::-1]
    for spec, col, mk, dy in SPECS:
        r = res[res.spec == spec].set_index("type")
        for t, yy in zip(types, ys):
            dot_ci(ax, yy + dy, r.loc[t, "est"], r.loc[t, "lo"], r.loc[t, "hi"], col, mk, filled=r.loc[t, "p"] < .05,
                   ms=4.2 if mk == "o" else 3.6)
    ax.axvline(0, color=SEC, lw=0.6, zorder=1)
    ax.set_yticks(ys, [t + ("†" if t == "作息紊亂型" else "") for t in types])
    ax.set_ylim(-0.6, len(types) - 0.4)
    ax.set_xlim(-2.6, 3.6)
    ax.set_xticks([-2, -1, 0, 1, 2, 3], [tick(v, "{:.0f}") for v in (-2, -1, 0, 1, 2, 3)])
    ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("國八憂鬱分數差（vs 早睡規律型；0–44 分）", labelpad=2)
    ax.text(3.5, ys[0] + 0.02, f"類型整體檢定\n{pfmt(chk['p_type_M2'])} → {pfmt(chk['p_type_given_bedtime'])}",
            ha="right", va="center", fontsize=7, color=INK, linespacing=1.15)
    ftext(fig, 0.15, H - 0.3, "(a) 加入國七上床時間後，類型差距縮小", fontsize=7.5, fontweight="bold", va="center")
    hs = [Line2D([], [], color=BLUE, marker="o", ms=4, lw=1.0, label="M2：控制國七憂鬱與背景"),
          Line2D([], [], color=NAVY, marker="D", ms=3.4, lw=1.0, label="＋國七上床時間"),
          Line2D([], [], color=SEC, marker="o", ms=4, mfc="white", ls="none", label="空心＝CI 含 0")]
    fig.legend(handles=hs, loc="center left", bbox_to_anchor=(0.15 / W, (H - 0.85) / H), ncol=3, handlelength=1.5,
               handletextpad=0.35, columnspacing=0.8, borderaxespad=0)
    # (b)
    ls_ = rd("mediation/tables/late_night_specificity.csv")
    ls_ = ls_[ls_["結果"] == "國八上床時間"].set_index("控制")
    bx = add_ax(fig, 11.05, 0.95, 5.75, 2.78)
    rows = [("醒著：以螢幕為主", "螢幕"), ("醒著：非螢幕活動", "非螢幕"), ("已入睡（參照）", None)]
    ys = np.arange(len(rows))[::-1]
    for (ctrl, col, mk, dy) in (("基本控制", BLUE, "o", 0.16), ("＋國七起床時間", NAVY, "D", -0.16)):
        r = ls_.loc[ctrl]
        for (lab, k), yy in zip(rows, ys):
            if k is None:
                continue
            dot_ci(bx, yy + dy, r[k], r[f"{k}_lo"], r[f"{k}_hi"], col, mk, filled=r[f"{k}_p"] < .05,
                   ms=4.2 if mk == "o" else 3.6)
    bx.plot(0, ys[-1], marker="o", ms=4.2, mfc=SEC, mec=SEC, ls="none", zorder=3)
    bx.axvline(0, color=SEC, lw=0.6, zorder=1)
    bx.set_yticks(ys, [r[0] for r in rows])
    bx.set_ylim(-0.6, len(rows) - 0.4)
    bx.set_xlim(-4, 30)
    bx.set_xticks([0, 10, 20, 30])
    bx.xaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    bx.set_axisbelow(True)
    bx.tick_params(axis="y", length=0)
    bx.spines["left"].set_visible(False)
    bx.set_xlabel("國八上床時間延後（分鐘）", labelpad=2)
    a, b = ls_.loc["基本控制"], ls_.loc["＋國七起床時間"]
    bx.text(29.5, ys[-1], f"螢幕 − 非螢幕：\n+{a['差']:.1f} 分鐘（{pfmt(a['差_p'])}）\n→ +{b['差']:.1f} 分鐘（{pfmt(b['差_p'])}）",
            ha="right", va="center", fontsize=7, linespacing=1.15)
    ftext(fig, 8.55, H - 0.3, "(b) 23 點後仍醒著：不用螢幕者一年後也更晚睡", fontsize=7.5, fontweight="bold",
          va="center")
    hs = [Line2D([], [], color=BLUE, marker="o", ms=4, lw=1.0, label="控制國七上床時間與憂鬱"),
          Line2D([], [], color=NAVY, marker="D", ms=3.4, lw=1.0, label="＋國七起床時間")]
    fig.legend(handles=hs, loc="center left", bbox_to_anchor=(8.55 / W, (H - 0.85) / H), ncol=2, handlelength=1.5,
               handletextpad=0.35, columnspacing=1.0, borderaxespad=0)
    save(fig, "fig4_not_only_screens.png")


# ====================================================================== 圖5
def fig_parenting():
    cm_ = rd("parenting/tables/coef_main.csv")
    cm_ = cm_[cm_["model"] == "主模型"].set_index(["outcome", "dim"])
    DIMS = [("restrict", "規範限制"), ("active", "陪伴引導"), ("encourage", "鼓勵上網"), ("worry", "擔憂叮嚀"),
            ("monitor", "監看"), ("conflict", "衝突責罵"), ("warmth", "溫暖支持"), ("hostile", "敵意／冷漠"),
            ("cohesion", "家庭凝聚")]
    OUTS = [("bedtime_2", "上床時間", "正值＝較晚睡", 1), ("sleep_hrs_2", "上課日睡眠時數", "正值＝睡較多", -1),
            ("addict_2", "網路成癮", "正值＝較嚴重", 1), ("cesd_nosleep_2", "憂鬱（不含睡眠題）", "正值＝較嚴重", 1)]
    ypos = np.array([9.6, 8.6, 7.6, 6.6, 5.6, 4.6, 2.9, 1.9, 0.9])  # 兩組之間留空
    H = 5.1
    fig = new_fig(H)
    left, right, gap, y0, ph = 2.2, 0.12, 0.28, 1.1, 3.05
    pw = (W - left - right - gap * 3) / 4
    XL = (-0.22, 0.30)
    for j, (o, title, hint, bad) in enumerate(OUTS):
        ax = add_ax(fig, left + j * (pw + gap), y0, pw, ph)
        ax.axvspan(0, XL[1], color="#f8eeea", zorder=0, lw=0) if bad > 0 else ax.axvspan(XL[0], 0, color="#f8eeea",
                                                                                         zorder=0, lw=0)
        for (k, _), yy in zip(DIMS, ypos):
            r = cm_.loc[(o, k)]
            col = BLUE if k not in ("warmth", "hostile", "cohesion") else SEC
            dot_ci(ax, yy, r.beta, r.lo, r.hi, col, "o", filled=r.q < .05, ms=3.8)
        ax.axvline(0, color=SEC, lw=0.6, zorder=1)
        ax.set_xlim(*XL)
        ax.set_ylim(0.3, 10.2)
        ax.set_xticks([-0.2, -0.1, 0, 0.1, 0.2, 0.3], [MINUS + ".2", "", "0", "", ".2", ""])
        ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_yticks(ypos, [d[1] for d in DIMS] if j == 0 else [])
        ax.text(0, 1.11, title, transform=ax.transAxes, fontsize=7.5, fontweight="bold", va="bottom")
        ax.text(0, 1.025, hint, transform=ax.transAxes, fontsize=7, color=SEC, va="bottom")
    # 分組標記：橫排兩行（不旋轉中文）
    fw, fh = fig.get_size_inches() / CM
    for lab, (top, bot), col in (("數位\n教養", (9.6, 4.6), BLUE), ("一般\n親子", (2.9, 0.9), SEC)):
        fy = lambda v: y0 + (v - 0.3) / (10.2 - 0.3) * ph  # noqa: E731
        ftext(fig, 0.3, (fy(top) + fy(bot)) / 2, lab, ha="center", va="center", fontsize=7.5, fontweight="bold",
              linespacing=1.05)
        fig.add_artist(Line2D([0.62 / fw] * 2, [(fy(top) + 0.12) / fh, (fy(bot) - 0.12) / fh], color=col, lw=0.8))
    ftext(fig, left + (W - left - right) / 2, 0.55, "標準化 β（95% CI）", ha="center", va="center", color=SEC)
    hs = [Line2D([], [], color=INK, marker="o", ms=3.8, ls="none", label="FDR q < .05"),
          Line2D([], [], color=INK, marker="o", ms=3.8, mfc="white", ls="none", label="未達顯著"),
          Patch(fc="#f8eeea", ec="none", label="對孩子較不利的方向")]
    fig.legend(handles=hs, loc="center", bbox_to_anchor=(0.5, 0.18 / H), ncol=3, handlelength=1.2,
               handletextpad=0.4, columnspacing=1.4, borderaxespad=0)
    save(fig, "fig5_parenting.png")


# ====================================================================== 表1
def table1():
    mp = rd("prediction/tables/model_performance.csv")
    mp = mp[mp["目標"] == "y_inc"].set_index("key")
    # 字串取自 outputs/prediction/results.md「### 新發生憂鬱或自傷意念風險…」表（第 33–39 行），由未四捨五入值格式化；
    # 下方以 model_performance.csv（四位小數）核對，避免二次四捨五入造成 ±0.001 的不一致。
    ROWS = [("cesd_only", "只看國七憂鬱分數（CESD）", "1", "0.671 [0.653, 0.687]", "0.358", "30.5%", "43.2%"),
            ("lr", "邏輯斯迴歸（L2）", "83", "0.718 [0.703, 0.734]", "0.412", "32.9%", "46.9%"),
            ("hgb", "梯度提升樹（HGB）", "83", "0.720 [0.704, 0.735]", "0.407", "32.5%", "46.3%"),
            ("rf", "隨機森林", "83", "0.722 [0.706, 0.738]", "0.404", "32.4%", "46.2%"),
            ("screener", "10 題簡短篩檢（邏輯斯）", "10", "0.706 [0.689, 0.724]", "0.388", "31.7%", "45.1%")]
    for key, _, _, auc, pr, se, ppv in ROWS:
        r = mp.loc[key]
        a, lo, hi = (float(v) for v in auc.replace("[", "").replace(",", "").replace("]", "").split())
        assert max(abs(a - r.AUC), abs(lo - r.AUC_lo), abs(hi - r.AUC_hi), abs(float(pr) - r.PR_AUC)) < .0011, key
        assert abs(float(se[:-1]) / 100 - r["敏感度@前15%"]) < .0011 and abs(float(ppv[:-1]) / 100 - r["PPV@前15%"]) < .0011
    n, prev = int(mp["n"].iloc[0]), mp["盛行率"].iloc[0]
    # 註腳 ΔAUC 取自 prediction/results.md 第 40–45 行（= tables/auc_differences.csv 之 y_inc 列）
    body = "\n".join(f"    <tr><td>{lab}</td><td>{k}</td><td>{auc}</td><td>{pr}</td><td>{se}</td><td>{ppv}</td></tr>"
                     for _, lab, k, auc, pr, se, ppv in ROWS)
    html = f"""<table class="table1">
  <caption>表1　以國七資料預測國八新發生憂鬱或自傷意念風險：五種模型的表現（n = {n:,}）</caption>
  <thead>
    <tr><th>模型</th><th>特徵數</th><th>ROC-AUC [95% CI]</th><th>PR-AUC</th><th>前 15% 敏感度</th><th>前 15% PPV</th></tr>
  </thead>
  <tbody>
{body}
  </tbody>
  <tfoot>
    <tr><td colspan="6">註：對象為國七 CESD &lt; 16 者；結果＝國八 CESD ≥ 16 或近 1–2 週自傷／死亡意念（加權盛行率 {prev:.1%}，即 PR-AUC 與 PPV 的隨機基準）。5 折「以學校分組」交叉驗證之樣本外預測，HOUWGT 加權；95% CI 為學校叢集 bootstrap。「前 15%」＝標記預測風險最高的 15% 學生。10 題（全樣本選題順序；交叉驗證時於各外折內另選）：滿意自己、覺得悲傷、無法上網就坐立不安、總是感到疲憊、覺得很寂寞、曾遭實體霸凌、女生、自評健康、睡眠情況不好、學校生活滿意。ΔAUC：隨機森林 − HGB +0.002 [−0.003, +0.007]；10 題篩檢 − 只看 CESD +0.036 [+0.022, +0.050]。</td></tr>
  </tfoot>
</table>
"""
    (TAB / "table1.html").write_text(html, encoding="utf-8")
    print("寫出", TAB / "table1.html")
    return html


if __name__ == "__main__":
    fig_hook()
    fig_types()
    fig_clpm()
    fig_twist()
    fig_parenting()
    table1()
