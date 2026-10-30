"""長版報告（report_long）的印刷尺寸圖表：由 outputs/*/tables 的彙總表重繪，不重新估計任何數值。

腳本 02–05 原本的圖以 10.5–16 英吋寬繪製，縮到 A4 版心（15.9 cm）後圖中文字只剩約 3–5 pt。
本腳本以版心寬度直接繪製（字級 6.5–7.5 pt，縮放比約 1.0），並刪去與圖說重複的圖內標題與註腳。

  fig_k_selection          typology/tables/k_selection.csv
  fig_type_timeline        typology/tables/type_hourly_composition.csv、type_sizes.csv
  fig_type_profiles        typology/tables/type_profiles_w1.csv（格內數值）；顏色所需的加權標準差由原始資料與
                           本機的類型歸屬檔（typology/tables/type_assignment.csv，個人層級、不公開）依腳本 02 同一定義
                           重算，並先核對各型加權平均與 type_profiles_w1.csv 一致（差異 > 1e-3 即中止）
  fig_type_outcomes        typology/tables/w2_outcomes_by_type.csv（M1、M2、M3）
  fig_mediation_bedtime／fig_mediation_sleephrs   mediation/tables/mediation_summary.csv（字串直接沿用，與表 12 一致）
  fig_indirect_forest      mediation/tables/mediation_paths.csv、sensitivity_exclude_highcesd.csv
  fig_collider_dag         示意圖；數值取自 parenting/tables/collider_decomposition.csv
  fig_rules_vs_scolding    parenting/tables/restrict_conflict_groups.csv
  fig_parenting_crosslag   parenting/tables/cross_lagged.csv、cross_lagged_path_difference.csv
  fig_restrict_adjustment  parenting/tables/restrict_adjustment_sets.csv
  fig_interaction          parenting/tables/interactions.csv（簡單斜率 ±1 SD 與 28 個交互作用項）
  fig_screener             prediction/tables/model_performance.csv、screener_auc_by_k.csv、screener_items.csv
  fig_shap                 prediction/tables/shap_importance_y_inc.csv、shap_domain_importance.csv
  fig_night_screen_prediction  prediction/tables/observed_rate_by_bin.csv、adjusted_or_sensitivity.csv、drop_domain_auc.csv
  fig_fairness             prediction/tables/fairness.csv
  fig_screen_vs_bedtime    screen_vs_bedtime/tables/coefficients.csv、difference_test.csv（模型代號同表 tab:screen_hours 的 ①–④）
  fig_clpm_sleephrs        mediation/tables/clpm_paths.csv（睡眠時數版；版面同腳本 06 的上床時間版 CLPM 圖）
  fig_prediction_performance   ROC／PR／校準曲線需要個人層級的樣本外預測（未輸出），故沿用腳本 05 的原圖，只裁去圖內標題
  fig_sequence_index       每列一位學生的序列（個人層級），沿用腳本 02 的原圖 typology/figures/fig_s2_sequence_index.png，只裁去圖內標題

執行：.venv/Scripts/python -I scripts/10_long_report_figures.py      輸出：report_long/figures/*.png（300 dpi）
"""
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")
OUT = ROOT / "outputs"
FIG = ROOT / "report_long" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
CM, W = 1 / 2.54, 15.9

INK, SEC, MUTED, GRID, BASE = "#1f1f1f", "#52514e", "#898781", "#e6e4de", "#c3c2b7"
BLUE, ORANGE, GREY, NAVY, RED = "#2a78d6", "#e3592f", "#8a8984", "#23255a", "#d9432f"
GREEN, AMBER, PURPLE, LIGHTBLUE = "#1baf7a", "#eda100", "#9b3a8f", "#9cc3ef"
DASH = "–"                       # Microsoft JhengHei 沒有 U+2212，負號以 en dash 表示（與腳本 06 相同）
plt.rcParams.update({
    "font.family": ["Microsoft JhengHei", "sans-serif"], "axes.unicode_minus": False, "font.size": 7,
    "axes.titlesize": 7.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 4,
    "axes.labelsize": 7, "xtick.labelsize": 6.8, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.linewidth": 0.5, "axes.edgecolor": BASE, "axes.labelcolor": SEC, "xtick.color": SEC, "ytick.color": SEC,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.dpi": 300, "legend.frameon": False, "axes.labelpad": 2,
    "hatch.linewidth": 0.5, "hatch.color": "white"})


def rd(p, **kw):
    return pd.read_csv(OUT / p, encoding="utf-8-sig", **kw)


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


def hu(v, d):
    """四捨五入（half-up），以 CSV 中的十進位字串為準，避免 0.6775 被二進位誤差捨成 0.677。"""
    return str(Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def num(v, d=2, plus=False):
    s = hu(abs(v), d)
    if v < 0 and float(s) != 0:
        return DASH + s
    return ("+" if plus and v > 0 and float(s) != 0 else "") + s


def pfmt(p):
    return "p < .001" if p < .001 else "p = " + hu(p, 3).replace("0.", ".", 1)


def axfmt(ax, axis="x", d=None):
    fmt = (lambda v, _: num(v, d) if d is not None else num(v, 0 if float(v).is_integer() else 2))
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(plt.FuncFormatter(fmt))


def dot(ax, y, est, lo, hi, color, marker="o", filled=True, ms=4.0, lw=1.0, horizontal=True, z=3):
    if horizontal:
        ax.hlines(y, lo, hi, color=color, lw=lw, zorder=z)
        ax.plot(est, y, marker=marker, ms=ms, mfc=color if filled else "white", mec=color, mew=0.9, ls="none", zorder=z + 1)
    else:
        ax.vlines(y, lo, hi, color=color, lw=lw, zorder=z)
        ax.plot(y, est, marker=marker, ms=ms, mfc=color if filled else "white", mec=color, mew=0.9, ls="none", zorder=z + 1)


def clean_y(ax):
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


TYPES = ["回家讀書型", "補習晚歸型", "電玩遊戲型", "影音社群夜貓型", "作息紊亂型"]       # 參照組＝早睡規律型
ORDER6 = ["早睡規律型"] + TYPES


# ====================================================================== 圖：選擇類別數
def fig_k_selection():
    k = rd("typology/tables/k_selection.csv")
    fig = new_fig(6.3)
    ks = k["k"].to_numpy()
    axes = [add_ax(fig, 0.95 + i * 5.1, 1.75, 3.9, 3.85) for i in range(3)]
    for ax in axes:
        ax.axvspan(5.6, 6.4, color="#fbe9b7", lw=0, zorder=0)
        ax.axvspan(6.6, 7.4, color="#ecebe6", lw=0, zorder=0)
        ax.set_xticks(ks)
        ax.set_xlim(2.6, 8.4)
        ax.set_xlabel("類別數 k")
        ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
    a = axes[0]
    for col, lab, c, m in (("ASW_LCA", "LCA", BLUE, "o"), ("ASW_PAM", "PAM", ORANGE, "s"), ("ASW_Ward", "Ward", GREEN, "^"),
                           ("ASW_kmodes", "k-modes", GREY, "D")):
        a.plot(ks, k[col], marker=m, ms=3.2, color=c, lw=1.0, label=lab, zorder=3)
    a.axhline(0.25, color=SEC, lw=0.6, ls=(0, (3, 2)))
    a.text(8.35, 0.253, "0.25", ha="right", va="bottom", fontsize=6.5, color=SEC)
    a.set_ylim(0.08, 0.27)
    a.set_ylabel("平均輪廓係數（ASW）")
    a.legend(loc="lower left", ncol=2, handlelength=1.4, columnspacing=0.8, borderaxespad=0.1)
    a.set_title("(a) 四種方法的 ASW")
    b = axes[1]
    drop = -np.diff(k["LCA_BIC"].to_numpy())
    b.bar(ks[1:], drop, width=0.6, color=BLUE, zorder=3)
    for x, v in zip(ks[1:], drop):
        b.text(x, v + 40, f"{v:,.0f}", ha="center", va="bottom", fontsize=6.5)
    b.set_ylim(0, 2400)
    b.set_ylabel("BIC 降幅（相對於 k − 1）")
    b.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    b.set_title("(b) LCA 每多一類的 BIC 降幅")
    c = axes[2]
    for col, lab, cl, m, ls in (("boot_ARI_lca", "LCA 平均 ARI", BLUE, "o", "-"), ("boot_minJaccard_lca", "LCA 最不穩一類 Jaccard", BLUE, "o", "--"),
                                ("boot_ARI_pam", "PAM 平均 ARI", ORANGE, "s", "-"), ("boot_minJaccard_pam", "PAM 最不穩一類 Jaccard", ORANGE, "s", "--")):
        c.plot(ks, k[col], marker=m, ms=3.0, color=cl, lw=1.0, ls=ls, label=lab, mfc=cl if ls == "-" else "white", zorder=3)
    for yv in (0.75, 0.5):
        c.axhline(yv, color=SEC, lw=0.6, ls=(0, (3, 2)))
    c.set_ylim(0.2, 1.03)
    c.set_ylabel("bootstrap 穩定度")
    h_, l_ = c.get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=6.4, handlelength=2.0,
               columnspacing=1.4, borderaxespad=0.2)
    c.set_title("(c) 學校叢集 bootstrap 穩定度")
    save(fig, "fig_k_selection.png")


# ====================================================================== 圖：六種晚上
STATES = ["補習課輔", "寫作業讀書", "生活與其他", "螢幕：影音", "螢幕：遊戲電視", "螢幕：社群聊天", "睡覺"]
STATE_LABEL = ["補習／課輔", "寫作業／讀書", "吃飯盥洗家事等生活", "螢幕：影音音樂", "螢幕：遊戲／電視", "螢幕：社群聊天", "睡覺"]
SCOL = ["#1baf7a", "#2a78d6", "#d9d5cb", "#f2b134", "#e3592f", "#a8326e", "#2e2a6b"]   # 與腳本 02 的序列索引圖相同


def fig_type_timeline():
    comp = rd("typology/tables/type_hourly_composition.csv", index_col=[0, 1])
    sizes = rd("typology/tables/type_sizes.csv").set_index("type")
    hrs = list(comp.columns)
    x = np.arange(len(hrs))
    fig = new_fig(10.6)
    pw, ph = 4.55, 3.0
    for i, t in enumerate(ORDER6):
        r_, c_ = divmod(i, 3)
        ax = add_ax(fig, 0.95 + c_ * (pw + 0.55), 6.0 - r_ * 4.45, pw, ph)
        bottom = np.zeros(len(hrs))
        for s, col in zip(STATES, SCOL):
            v = comp.loc[(t, s)].to_numpy(float)
            ax.bar(x, v, bottom=bottom, width=0.86, color=col, edgecolor="white", linewidth=0.3, zorder=2)
            bottom += v
        ax.plot(x, comp.loc[(t, "任何螢幕（主要或次要）")].to_numpy(float), color=INK, lw=0.9, marker="o", ms=2.6,
                mfc="white", mec=INK, mew=0.7, zorder=4)
        ax.set_xlim(-0.55, len(hrs) - 0.45)
        ax.set_ylim(0, 1)
        ax.set_xticks(x, [h.split(":")[0] for h in hrs])
        ax.set_yticks([0, .5, 1], ["0", "50%", "100%"] if c_ == 0 else [])
        for sp in ("left", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)
        r = sizes.loc[t]
        aw = "<1%" if r.awake_midnight_pct < 1 else f"{hu(r.awake_midnight_pct, 0)}%"
        ax.text(0, 1.17, t + ("†" if t == "作息紊亂型" else ""), transform=ax.transAxes, fontsize=7.5, fontweight="bold")
        ax.text(0, 1.04, f"占 {hu(r.pct_w, 1)}%（n = {int(r.n):,}）｜午夜仍醒 {aw}", transform=ax.transAxes, fontsize=6.6,
                color=SEC)
        if r_ == 1:
            ax.set_xlabel("時段起點（時；0＝午夜 0:00–1:00）", fontsize=6.6)
    hs = [Patch(fc=c, ec="none", label=l) for c, l in zip(SCOL, STATE_LABEL)]
    hs.append(Line2D([], [], color=INK, lw=0.9, marker="o", ms=2.6, mfc="white", mec=INK, label="任何螢幕（主要或次要活動）"))
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, handlelength=1.2, handleheight=0.8,
               columnspacing=1.0, handletextpad=0.4, borderaxespad=0.2)
    save(fig, "fig_type_timeline.png")


# ====================================================================== 圖：各型國七輪廓
PROF = [("female", "女生 %", 100, 0), ("cram", "有補習 %", 100, 0), ("digi_mature", "數位成熟區 %", 100, 0),
        ("private", "私立學校 %", 100, 0), ("fam_econ", "家庭經濟（1–4，大＝富裕）", 1, -1),
        ("grade_rank", "成績排名（1–5，大＝差）", 1, 1), ("net_leisure", "休閒上網指數（時）", 1, 1),
        ("addict", "網路成癮（1–4）", 1, 1), ("fomo", "FOMO（1–4）", 1, 1), ("bedtime", "自填上床時間（時）", 1, 1),
        ("sleep_hrs", "自填睡眠時數", 1, -1), ("cesd", "憂鬱 CESD（0–48）", 1, 1)]


def profile_sds():
    """依腳本 02 同一定義重算各輪廓變項在課堂 A 卷樣本的加權 SD，並核對各型加權平均。"""
    from tigps import data as D
    raw1, _ = D.read_wave(D.W1_SAV)
    w1 = D.build_w1(raw1)
    ta = rd("typology/tables/type_assignment.csv")
    ta = ta[ta["randA"].astype(bool)]
    A = w1[w1["has_A"]].merge(ta[["nstudent_id", "type"]], on="nstudent_id", how="inner")
    A["digi_mature"] = (A["digi_level"] == "數位發展成熟區").astype(float)
    A["private"] = (A["school_type"] == 2).astype(float)
    ref = rd("typology/tables/type_profiles_w1.csv", index_col=0)
    sds, worst = {}, 0.0
    for v, lbl, mult, _ in PROF:
        for t in ORDER6:
            g = A[(A["type"] == t)][[v, "HOUWGT"]].dropna()
            m = mult * np.average(g[v], weights=g["HOUWGT"])
            key = [ix for ix in ref.index if ix.split("（")[0].split(" %")[0] == lbl.split("（")[0].split(" %")[0]][0]
            worst = max(worst, abs(m - ref.loc[key, t]))
        d = A[[v, "HOUWGT"]].dropna()
        sds[lbl] = mult * np.sqrt(np.cov(d[v], aweights=d["HOUWGT"]))
    if len(A) != 4335 or worst > 1e-3:
        sys.exit(f"類型輪廓核對失敗：n = {len(A)}，最大平均差 {worst:.5f}")
    print(f"  輪廓核對：n = {len(A)}，各型加權平均與 type_profiles_w1.csv 最大差 {worst:.1e}")
    return sds, ref


def fig_type_profiles():
    sds, ref = profile_sds()
    keys = list(ref.index)
    vals = np.array([[ref.loc[k_, t] for t in ORDER6] for k_ in keys])
    allv = ref["全體"].to_numpy()
    sd = np.array([sds[lbl] for _, lbl, _, _ in PROF])
    z = (vals - allv[:, None]) / sd[:, None]
    dirs = [d for *_, d in PROF]
    div = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f3f2ee", "#e3592f"])
    neu = LinearSegmentedColormap.from_list("neu", ["#f3f2ee", "#a5a196"])
    rgba = np.array([div((dr * z[i].clip(-1, 1) + 1) / 2) if dr else neu(np.abs(z[i]).clip(0, 1))
                     for i, dr in enumerate(dirs)])
    fig = new_fig(9.0)
    ax = add_ax(fig, 4.7, 0.25, W - 4.85, 7.9)
    ax.imshow(rgba, aspect="auto")
    for i, (_, lbl, _, _) in enumerate(PROF):
        for j in range(len(ORDER6)):
            v = vals[i, j]
            txt = hu(v, 0) if "%" in lbl else (hu(v, 1) if abs(v) >= 10 else hu(v, 2))
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if dirs[i] and abs(z[i, j]) > 0.75 else INK)
    ax.set_xticks(range(len(ORDER6)), [t + ("†" if t == "作息紊亂型" else "") for t in ORDER6], fontsize=7)
    ax.xaxis.tick_top()
    ylab = [f"{lbl}  [{hu(a, 0)}]" if "%" in lbl else f"{lbl}  [{hu(a, 2)}]" for (_, lbl, _, _), a in zip(PROF, allv)]
    ax.set_yticks(range(len(PROF)), ylab, fontsize=7)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(ORDER6)), minor=True)
    ax.set_yticks(np.arange(-.5, len(PROF)), minor=True)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.axhline(3.5, color=INK, lw=0.8)
    save(fig, "fig_type_profiles.png")


# ====================================================================== 圖：類型與一年後結果
OUTC = [("cesd_nosleep_2", "國八憂鬱（不含睡眠題）", "分，0–44；正值較差", 1),
        ("selfharm_2pp", "國八自傷／死亡意念", "百分點；正值較差", 1),
        ("bedtime_2m", "國八上床時間", "分鐘；正值較晚", 1),
        ("sleep_hrs_2m", "國八睡眠時數", "分鐘；正值較多", 1),
        ("addict_2", "國八網路成癮", "分，1–4；正值較差", 1),
        ("grade_rank_2", "國八成績排名", "級距（大＝差）", 1),
        ("who5_2", "國八幸福感 WHO-5", "分，0–20；正值較好", 1)]
SPEC = [("M1", GREY, "D", 0.26, 2.8), ("M2", BLUE, "o", 0.0, 3.6), ("M3", ORANGE, "s", -0.26, 3.0)]


def fig_type_outcomes():
    t = rd("typology/tables/w2_outcomes_by_type.csv")
    t = t[t["sample"] == "main"]
    fig = new_fig(11.0)
    pw, ph, x0 = 2.95, 3.95, 2.25
    ys = np.arange(len(TYPES))[::-1]
    for i, (oc, title, unit, _) in enumerate(OUTC):
        r_, c_ = divmod(i, 4)
        ax = add_ax(fig, x0 + c_ * (pw + 0.42), 6.25 - r_ * 5.3, pw, ph)
        sub = t[t.outcome == oc]
        for spec, col, mk, dy, ms in SPEC:
            s = sub[sub.spec == spec].set_index("type")
            for ty, yy in zip(TYPES, ys):
                r = s.loc[ty]
                dot(ax, yy + dy, r.est, r.ci_lo, r.ci_hi, col, mk, filled=r.q_fdr < .05, ms=ms, lw=0.9)
        s2 = sub[sub.spec == "M2"].set_index("type")
        xl = ax.get_xlim()
        span = xl[1] - xl[0]
        for ty, yy in zip(TYPES, ys):
            r = s2.loc[ty]
            if r.q_fdr < .05:
                d = 1 if abs(r.est) >= 1 else 2
                if r.est >= 0:
                    ax.text(r.ci_hi + 0.03 * span, yy + 0.02, num(r.est, d, plus=True), fontsize=6.3, color=BLUE, va="center")
                else:
                    ax.text(r.ci_lo - 0.03 * span, yy + 0.02, num(r.est, d), fontsize=6.3, color=BLUE, va="center", ha="right")
        ax.set_xlim(xl[0] - 0.1 * span, xl[1] + 0.12 * span)
        ax.axvline(0, color=SEC, lw=0.6, zorder=1)
        ax.set_ylim(-0.6, len(TYPES) - 0.4)
        ax.set_yticks(ys, TYPES if c_ == 0 else [])
        axfmt(ax, "x")
        ax.tick_params(axis="x", labelsize=6.5)
        clean_y(ax)
        ax.set_title(title, fontsize=7.2)
        ax.set_xlabel(unit, fontsize=6.5)
    hs = [Line2D([], [], color=c, marker=m, ms=ms, lw=0.9, label=l) for (s, c, m, _, ms), l in
          zip(SPEC, ["M1：控制性別、家庭經濟、\n數位發展分區", "M2：再控制國七同一指標\n（主要結果）",
                     "M3：再控制國七身心、\n學校與家庭變項"])]
    hs.append(Line2D([], [], color=SEC, marker="o", ms=3.4, mfc="white", ls="none", label="空心＝BH-FDR 校正後\n不顯著"))
    hs.append(Line2D([], [], color=BLUE, marker="", ls="none", label="藍字＝M2 中 q < .05\n的估計值"))
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=((x0 + 3 * (pw + 0.42) - 0.1) / W, 5.0 / 11.0), fontsize=6.4,
               handlelength=1.6, labelspacing=0.9, borderaxespad=0)
    save(fig, "fig_type_outcomes.png")


# ====================================================================== 圖：半縱貫中介路徑
def fig_mediation(mname, fname):
    ms = rd("mediation/tables/mediation_summary.csv")
    ms = ms[ms["M"] == mname].set_index("X")
    rows = [("網路成癮（A 卷）", "國七\n網路成癮", "網路成癮"), ("23 點後仍醒著（A 卷）", "國七 23 點後\n仍醒著", "熬夜：23 點後仍醒著（vs 已入睡）"),
            ("FOMO（A 卷）", "國七\nFOMO", "FOMO 錯失恐懼")]
    fig = new_fig(6.9)
    bx = add_ax(fig, 0, 0, W, 6.9)
    bx.set_xlim(0, W)
    bx.set_ylim(0, 6.9)
    bx.axis("off")
    bw, bh = 1.55, 0.82

    def box(x, y, s):
        bx.add_patch(FancyBboxPatch((x - bw / 2, y - bh / 2), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="#eef3fb", ec=NAVY, lw=0.7))
        bx.text(x, y, s, ha="center", va="center", fontsize=7, linespacing=1.15)

    def sig(sv):
        return "*" in sv

    def arrow(p0, p1, solid, color):
        bx.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, lw=1.0 if solid else 0.8,
                                     color=color if solid else NS_C, ls="-" if solid else (0, (3, 2)), shrinkA=0, shrinkB=0))

    NS_C = "#b5b3ad"
    mlab = "國八\n上床時間" if mname == "上床時間" else "國八\n睡眠時數"
    for i, (key, xlab, title) in enumerate(rows):
        r = ms.loc[key]
        cx = 0.25 + i * 5.25
        xl, xr, xm = cx + 0.95, cx + 3.95, cx + 2.45
        yb, yt = 3.15, 5.35
        bx.text(cx + 2.55, 6.6, f"({'abc'[i]}) {title}", ha="center", va="center", fontsize=7.3, fontweight="bold")
        box(xl, yb, xlab)
        box(xr, yb, "國八\n憂鬱症狀")
        box(xm, yt, mlab)
        a, b, c, cp = r["a (β)"], r["b 同期 (β)"], r["c 總效果 (β)"], r["c' 直接 (β)"]
        arrow((xl + 0.2, yb + bh / 2), (xm - 0.4, yt - bh / 2), sig(a), BLUE)
        arrow((xm + 0.4, yt - bh / 2), (xr - 0.2, yb + bh / 2), sig(b), BLUE)
        arrow((xl + bw / 2, yb), (xr - bw / 2, yb), sig(cp), SEC)
        bx.text((xl + xm) / 2 - 0.24, (yb + yt) / 2 + 0.05, f"a = {a.replace('-', DASH)}", ha="right", va="center", fontsize=6.6,
                color=BLUE if sig(a) else MUTED)
        bx.text((xm + xr) / 2 + 0.24, (yb + yt) / 2 + 0.05, f"b = {b.replace('-', DASH)}", ha="left", va="center", fontsize=6.6,
                color=BLUE if sig(b) else MUTED)
        bx.text(xm, yb - 0.75, f"c′ = {cp.replace('-', DASH)}（總關聯 c = {c.replace('-', DASH)}）", ha="center", va="center",
                fontsize=6.5, color=SEC)
        ab, q = r["a×b (β) [95% CI]"], r["q"]
        qs = "q < .001" if q < .001 else f"q = {q:.3f}".replace("0.", ".", 1)
        bx.text(xm, yb - 1.35, f"間接關聯 a × b = {ab.replace('-', DASH)}", ha="center", va="center", fontsize=6.5,
                color=BLUE if q < .05 else SEC, fontweight="bold" if q < .05 else "normal")
        bx.text(xm, yb - 1.85, f"{qs}；n = {int(r['n（校）'].split('（')[0]):,}（{r['n（校）'].split('（')[1].rstrip('）')} 校）", ha="center", va="center",
                fontsize=6.6, color=SEC)
        pm_t, pm_c = r["中介比例 時序版 [CI]"], r["中介比例 同期版 [CI]"]
        if isinstance(pm_c, str) and pm_c != "—":
            bx.text(xm, yb - 2.33, f"中介比例：時序版 {pm_t.split(' ')[0]}、同期版 {pm_c.split(' ')[0]}", ha="center",
                    va="center", fontsize=6.6, color=SEC)
        else:
            bx.text(xm, yb - 2.33, "總關聯不顯著，不計中介比例", ha="center", va="center", fontsize=6.6, color=SEC)
    save(fig, fname)


# ====================================================================== 圖：間接關聯森林圖
def fig_indirect_forest():
    mp = rd("mediation/tables/mediation_paths.csv")
    ex = rd("mediation/tables/sensitivity_exclude_highcesd.csv")
    rows = [("A 卷", "addict", "網路成癮"), ("全樣本", "addict", "網路成癮（全樣本†）"), ("A 卷", "awake", "23 點後仍醒著"),
            ("A 卷", "scr", "其中：以螢幕為主 vs 已入睡"), ("A 卷", "oth", "其中：只做非螢幕 vs 已入睡"),
            ("A 卷", "dif", "螢幕 − 非螢幕".replace("−", DASH)), ("A 卷", "fomo", "FOMO")]
    fig = new_fig(8.1)
    ys = np.arange(len(rows))[::-1]
    est = [("ab_std", "ab_std_lo", "ab_std_hi", "主分析（同期 b）", BLUE, "o", 0.24, mp),
           ("ab_cm_std", "ab_cm_std_lo", "ab_cm_std_hi", "Cole–Maxwell 時序版", ORANGE, "s", 0.0, mp),
           ("ab_std", "ab_std_lo", "ab_std_hi", "排除國七高憂鬱症狀者（同期 b）", GREEN, "^", -0.24, ex)]
    for j, (mname, title) in enumerate((("上床時間", "(a) 中介＝國八上床時間"), ("睡眠時數", "(b) 中介＝國八睡眠時數"))):
        ax = add_ax(fig, 4.15 + j * 6.0, 1.05, 5.4, 5.75)
        ax.axhspan(ys[5] - 0.5, ys[3] + 0.5, color="#f1f0ec", lw=0, zorder=0)
        for col, lo, hi, lab, c, m, dy, src in est:
            s = src[(src["M"] == mname)]
            for (smp, key, _), yy in zip(rows, ys):
                r = s[(s["樣本"] == smp) & (s["key"] == key)]
                if r.empty or pd.isna(r.iloc[0][col]):
                    continue
                r = r.iloc[0]
                dot(ax, yy + dy, r[col], r[lo], r[hi], c, m, ms=3.2, lw=0.9)
        ax.axvline(0, color=SEC, lw=0.6, zorder=1)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_yticks(ys, [r[2] for r in rows] if j == 0 else [])
        ax.set_xlim(-0.006, 0.033)
        ax.set_xticks([0, 0.01, 0.02, 0.03])
        axfmt(ax, "x", 2)
        clean_y(ax)
        ax.set_title(title)
        ax.set_xlabel("完全標準化間接關聯 a × b（95% CI）")
    hs = [Line2D([], [], color=c, marker=m, ms=3.2, lw=0.9, label=lab) for _, _, _, lab, c, m, _, _ in est]
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.01, 0.995), ncol=3, columnspacing=1.2, handlelength=1.6,
               borderaxespad=0.1)
    save(fig, "fig_indirect_forest.png")


# ====================================================================== 圖：對撞因子示意
def fig_collider_dag():
    cd = rd("parenting/tables/collider_decomposition.csv").set_index("outcome").loc["addict_2"]
    H, Wd = 5.6, 12.0
    fig = new_fig(H, Wd)
    bx = add_ax(fig, 0, 0, Wd, H)
    bx.set_xlim(0, Wd)
    bx.set_ylim(0, H)
    bx.axis("off")

    def node(x, y, s, fc="#eef3fb", ec=NAVY, w=2.9, h=0.95, lw=0.7):
        bx.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc,
                                    ec=ec, lw=lw))
        bx.text(x, y, s, ha="center", va="center", fontsize=7.2, linespacing=1.2)

    node(2.0, 4.3, "國八規範限制\n（規則）")
    node(10.0, 4.3, "孩子問題上網\n（網路成癮）")
    node(6.0, 1.55, "衝突責罵\n（對撞因子）", fc="#fdeee6", ec=ORANGE, lw=1.1)
    for p0, p1 in (((2.6, 3.8), (5.0, 2.05)), ((9.4, 3.8), (7.0, 2.05))):
        bx.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, lw=1.0, color=INK))
    bx.add_patch(FancyArrowPatch((3.5, 4.3), (8.5, 4.3), arrowstyle="-", lw=0.9, color=RED, ls=(0, (3, 2))))
    bx.text(6.0, 4.55, "假設規則對成癮毫無作用", ha="center", va="bottom", fontsize=6.8, color=SEC)
    bx.text(6.0, 4.05, f"只看兩者：r = {num(cd['r(規則,Y)'], 3, plus=True).replace('0.', '.', 1)}", ha="center", va="top",
            fontsize=6.8, color=SEC)
    bx.text(6.0, 3.3, f"控制衝突責罵後：β = {num(cd['β規則|衝突'], 3)}\n（虛假的負相關）", ha="center", va="top", fontsize=7,
            color=RED, fontweight="bold", linespacing=1.3)
    bx.text(2.0, 2.55, "規則越多、越常起衝突\n" + f"r(規則, 衝突) = {cd['r(規則,衝突)']:.3f}".replace("0.", ".", 1),
            ha="center", va="center", fontsize=6.6, color=SEC, linespacing=1.2)
    bx.text(10.0, 2.55, "孩子越常上網、\n越常因此被責罵", ha="center", va="center", fontsize=6.6, color=SEC, linespacing=1.2)
    bx.text(6.0, 0.45, "以對撞因子為條件（放進模型控制），會在原本無關的兩者之間製造關聯", ha="center", va="center",
            fontsize=6.6, color=SEC)
    save(fig, "fig_collider_dag.png")


# ====================================================================== 圖：規則 × 責罵 四格
def fig_rules_vs_scolding():
    g = rd("parenting/tables/restrict_conflict_groups.csv")
    groups = ["有規則・幾乎不責罵", "少規則・幾乎不責罵", "有規則・有責罵", "少規則・有責罵"]
    labels = ["有規則\n幾乎\n不責罵", "少規則\n幾乎\n不責罵", "有規則\n有責罵", "少規則\n有責罵"]
    cols = [BLUE, LIGHTBLUE, ORANGE, "#f4b09a"]
    panels = [("午夜(含)後才睡 %", "(a) 午夜（含）以後才睡（%）", 1, (0, 40)), ("網路成癮 (1–4)", "(b) 網路成癮（1–4）", 2, (1, 2.5)),
              ("憂鬱 CESD（不含睡眠題）", "(c) 憂鬱（不含睡眠題，0–44）", 2, (0, 10))]
    fig = new_fig(5.8)
    for i, (var, title, d, yl) in enumerate(panels):
        ax = add_ax(fig, 0.75 + i * 5.15, 1.45, 4.3, 3.7)
        s = g[g["變項"] == var].set_index("組別").loc[groups]
        x = np.arange(4)
        ax.bar(x, s["mean"] - yl[0], bottom=yl[0], width=0.62, color=cols, zorder=2)
        ax.errorbar(x, s["mean"], yerr=[s["mean"] - s["lo"], s["hi"] - s["mean"]], fmt="none", ecolor=INK, elinewidth=0.6,
                    capsize=1.5, capthick=0.6, zorder=3)
        for xx, v, h in zip(x, s["mean"], s["hi"]):
            ax.text(xx, h + (yl[1] - yl[0]) * 0.02, hu(v, d), ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x, labels, fontsize=6.4, linespacing=1.05)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(*yl)
        ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(title)
    save(fig, "fig_rules_vs_scolding.png")


# ====================================================================== 圖：教養交叉延宕
DIMS = [("restrict", "規範限制"), ("active", "陪伴引導"), ("encourage", "鼓勵上網"), ("worry", "擔憂叮嚀"),
        ("monitor", "監看"), ("conflict", "衝突責罵")]


def fig_parenting_crosslag():
    cl = rd("parenting/tables/cross_lagged.csv")
    pdf = rd("parenting/tables/cross_lagged_path_difference.csv")
    fig = new_fig(6.8)
    ys = np.arange(len(DIMS))[::-1]
    for j, (oc, title) in enumerate((("網路成癮", "(a) 數位教養與網路成癮"), ("憂鬱", "(b) 數位教養與憂鬱"))):
        ax = add_ax(fig, 1.55 + j * 7.25, 1.55, 5.2, 4.55)
        for (dk, _), yy in zip(DIMS, ys):
            for dirn, col, mk, dy in (("親→子", BLUE, "o", 0.16), ("子→親", ORANGE, "s", -0.16)):
                r = cl[(cl["方向"] == dirn) & (cl["教養"] == dk) & (cl["結果"] == oc)].iloc[0]
                dot(ax, yy + dy, r.beta, r.lo, r.hi, col, mk, filled=r.q < .05, ms=3.4, lw=0.9)
            pr = pdf[(pdf["教養"] == dk) & (pdf["結果"] == oc)].iloc[0]
            ax.text(1.02, yy, f"差 {pfmt(pr.p)}", transform=ax.get_yaxis_transform(), fontsize=6.4, va="center",
                    color=INK if pr.q < .05 else MUTED)
        ax.axvline(0, color=SEC, lw=0.6, zorder=1)
        ax.set_ylim(-0.6, len(DIMS) - 0.4)
        ax.set_yticks(ys, [d[1] for d in DIMS] if j == 0 else [])
        ax.set_xlim(-0.08, 0.11)
        ax.set_xticks([-0.05, 0, 0.05, 0.1])
        axfmt(ax, "x", 2)
        clean_y(ax)
        ax.set_title(title)
        ax.set_xlabel("標準化交叉延宕係數（95% CI）")
    hs = [Line2D([], [], color=BLUE, marker="o", ms=3.4, lw=0.9, label="親→子：國七教養 → 國八孩子（控制國七孩子）"),
          Line2D([], [], color=ORANGE, marker="s", ms=3.4, lw=0.9, label="子→親：國七孩子 → 國八教養（控制國七教養）"),
          Line2D([], [], color=SEC, marker="o", ms=3.4, mfc="white", ls="none", label="空心＝BH-FDR 校正後不顯著；右側黑字＝兩方向差異 q < .05")]
    fig.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=2, handlelength=1.6, columnspacing=1.2,
               borderaxespad=0.15, fontsize=6.4)
    save(fig, "fig_parenting_crosslag.png")


# ====================================================================== 圖：規範限制與調整集合
def fig_restrict_adjustment():
    t = rd("parenting/tables/restrict_adjustment_sets.csv")
    sets = list(dict.fromkeys(t["調整集合"]))
    lab = {"僅共變項": "只控共變項", "主模型：+一般親子": "主模型：＋一般親子", "+一般親子+陪伴/鼓勵": "＋一般親子＋陪伴／鼓勵",
           "+一般親子+擔憂叮嚀": "＋一般親子＋擔憂叮嚀", "+一般親子+監看": "＋一般親子＋監看",
           "+一般親子+衝突責罵": "＋一般親子＋衝突責罵", "全部互相控制": "六構面全部互相控制"}
    outs = [("bedtime_2", "(a) 上床時間"), ("sleep_hrs_2", "(b) 睡眠時數"), ("addict_2", "(c) 網路成癮"),
            ("cesd_nosleep_2", "(d) 憂鬱（不含睡眠題）")]
    fig = new_fig(6.0)
    ys = np.arange(len(sets))[::-1]
    for j, (oc, title) in enumerate(outs):
        ax = add_ax(fig, 3.35 + j * 3.15, 1.6, 2.7, 3.85)
        s = t[t.outcome == oc].set_index("調整集合")
        for st, yy in zip(sets, ys):
            r = s.loc[st]
            col = BLUE if st.startswith("主模型") else (ORANGE if any(k_ in st for k_ in ("擔憂", "監看", "衝突", "全部")) else GREY)
            dot(ax, yy, r.beta, r.lo, r.hi, col, "o", filled=r.p < .05, ms=3.4, lw=0.9)
        ax.axvline(0, color=SEC, lw=0.6, zorder=1)
        ax.set_ylim(-0.6, len(sets) - 0.4)
        ax.set_yticks(ys, [lab.get(s_, s_) for s_ in sets] if j == 0 else [])
        lo_, hi_ = (-0.25, 0.05) if oc == "bedtime_2" else ((-0.05, 0.25) if oc == "sleep_hrs_2" else (-0.2, 0.08))
        ax.set_xlim(lo_, hi_)
        axfmt(ax, "x", 1)
        clean_y(ax)
        ax.set_title(title, fontsize=7.2)
        ax.set_xlabel("規範限制的標準化 β", fontsize=6.5)
    ftext(fig, 0.15, 0.2, "藍＝主模型；橘＝加入可能是孩子問題結果的構面（對撞因子疑慮）；空心＝p 未達 .05", fontsize=6.4, color=SEC)
    save(fig, "fig_restrict_adjustment.png")


# ====================================================================== 圖：調節分析
MODS = [("dp_restrict", "規範限制"), ("dp_active", "陪伴引導"), ("dp_encourage", "鼓勵上網"), ("dp_worry", "擔憂叮嚀"),
        ("dp_monitor", "監看"), ("dp_conflict", "衝突責罵"), ("warmth", "溫暖支持")]


def fig_interaction():
    it = rd("parenting/tables/interactions.csv")
    fig = new_fig(9.7)
    # (a) 同期：成癮 → 憂鬱的簡單斜率（調節變項 −1 SD vs +1 SD）
    ax = add_ax(fig, 1.6, 1.65, 5.0, 7.5)
    s = it[(it["類型"] == "同期") & (it["X"] == "addict_2")]
    ys = np.arange(len(MODS))[::-1]
    for (mk, lab), yy in zip(MODS, ys):
        r = s[s["M"] == f"{mk}_2"].iloc[0]
        for col, k_, dy, m in ((LIGHTBLUE, "M-1SD", 0.15, "o"), (NAVY, "M+1SD", -0.15, "D")):
            v, se = r[f"slope_{k_}"], r[f"se_{k_}"]
            dot(ax, yy + dy, v, v - 1.96 * se, v + 1.96 * se, col, m, ms=3.4, lw=0.9)
        ax.text(1.02, yy, ("q = " + hu(r.q, 3).replace("0.", ".", 1)) if r.q >= .001 else "q < .001",
                transform=ax.get_yaxis_transform(), fontsize=6.4, va="center", color=INK if r.q < .05 else MUTED)
    ax.set_ylim(-0.6, len(MODS) - 0.4)
    ax.set_yticks(ys, [m[1] for m in MODS])
    ax.set_xlim(0.08, 0.34)
    axfmt(ax, "x", 2)
    clean_y(ax)
    ax.set_title("(a) 國八成癮 → 國八憂鬱的標準化斜率")
    ax.set_xlabel("簡單斜率（95% CI）")
    hs = [Line2D([], [], color=LIGHTBLUE, marker="o", ms=3.4, lw=0.9, label="調節變項低（−1 SD）".replace("−", DASH)),
          Line2D([], [], color=NAVY, marker="D", ms=3.4, lw=0.9, label="調節變項高（+1 SD）")]
    ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=2, fontsize=6.3, borderaxespad=0.1)
    # (b) 28 個交互作用項
    BX0, BW = 10.9, 4.6
    bx = add_ax(fig, BX0, 1.65, BW, 7.5)
    groups = [("同期", "addict_2", "同期：成癮 × 教養"), ("同期", "bedtime_2", "同期：上床時間 × 教養"),
              ("半前瞻(W1成癮×W2教養)", "addict_1", "半前瞻：國七成癮 × 國八教養"), ("W1前瞻(B卷)", "addict_1", "國七前瞻：成癮 × 教養（B 卷）")]
    y, yt, ylab, gpos = 0, [], [], []
    rows = []
    for typ, x, gl in groups:
        sub = it[(it["類型"] == typ) & (it["X"] == x)]
        if rows:
            y += 0.35   # 組間多留空，組標題白底不蓋到上一組最後一列的標籤
        gpos.append((y, gl))
        y += 1
        for mk, lab in MODS:
            r = sub[sub["M"].str.startswith(mk)].iloc[0]
            rows.append((y, r))
            yt.append(y)
            ylab.append(lab)
            y += 1
    for yy, r in rows:
        dot(bx, -yy, r.beta_int, r.lo, r.hi, ORANGE if r.q < .05 else BLUE, "o", filled=r.q < .05, ms=3.0, lw=0.8)
    for yy, gl in gpos:   # 組標題放在左側標籤欄（自 8.0 cm 起），白底遮住格線，不與零線相交
        bx.text((8.0 - BX0) / BW, -yy, gl, transform=bx.get_yaxis_transform(), fontsize=6.5, fontweight="bold",
                va="center", ha="left", zorder=5, bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))
    bx.axvline(0, color=SEC, lw=0.6, zorder=1)
    bx.set_ylim(-y + 0.4, 0.6)
    bx.set_yticks([-v for v in yt], ylab, fontsize=6.1)
    bx.set_xlim(-0.1, 0.11)
    axfmt(bx, "x", 2)
    clean_y(bx)
    bx.set_title("(b) 28 個交互作用項 β（95% CI）")
    bx.set_xlabel("交互作用項（負值＝關聯被減弱）")
    save(fig, "fig_interaction.png")


# ====================================================================== 圖：10 題簡短篩檢
MODEL_LAB = {"cesd_only": "只用國七 CESD", "screener": "10 題簡短篩檢", "lr": "L2 邏輯斯迴歸", "rf": "隨機森林", "hgb": "梯度提升樹（HGB）"}
MCOL = {"cesd_only": GREY, "screener": "#eb6834", "lr": GREEN, "rf": AMBER, "hgb": BLUE}   # 與腳本 05 的圖（圖 fig_prediction_performance）同色


def fig_screener():
    mp = rd("prediction/tables/model_performance.csv")
    mp = mp[mp["目標"] == "y_inc"].set_index("key")
    cv = rd("prediction/tables/screener_auc_by_k.csv")
    cv = cv[cv["目標"] == "y_inc"]
    it = rd("prediction/tables/screener_items.csv")
    it = it[it["目標"] == "y_inc"].sort_values("入選順序")
    fig = new_fig(8.6)
    a = add_ax(fig, 2.45, 5.25, 4.6, 2.75)
    order = ["cesd_only", "screener", "lr", "rf", "hgb"]
    for k_, m in enumerate(order):
        r = mp.loc[m]
        dot(a, k_, r.AUC, r.AUC_lo, r.AUC_hi, MCOL[m], "o", ms=3.6, lw=1.0)
        a.text(r.AUC_hi + 0.003, k_, hu(r.AUC, 3), va="center", fontsize=6.5)
    a.set_yticks(range(len(order)), [MODEL_LAB[m] for m in order])
    a.set_ylim(-0.6, len(order) - 0.4)
    a.set_xlim(0.645, 0.76)
    axfmt(a, "x", 2)
    clean_y(a)
    a.set_title("(a) 五種模型的 ROC-AUC（95% CI）")
    b = add_ax(fig, 1.0, 0.95, 6.05, 3.15)
    b.fill_between(cv["題數"], cv["CI下"], cv["CI上"], color=MCOL["screener"], alpha=0.18, lw=0)
    b.plot(cv["題數"], cv["AUC"], "o-", color=MCOL["screener"], lw=1.0, ms=3.0, mec="white", mew=0.5, label="簡短篩檢（各外折內選題）")
    b.axhline(mp.loc["hgb", "AUC"], color=BLUE, ls="--", lw=0.8, label=f"HGB 全特徵 {hu(mp.loc['hgb', 'AUC'], 3)}")
    b.axhline(mp.loc["cesd_only", "AUC"], color=GREY, ls=":", lw=0.9, label=f"只用國七 CESD {hu(mp.loc['cesd_only', 'AUC'], 3)}")
    b.set_xticks(range(1, 11))
    b.set_xlabel("篩檢題數")
    b.set_ylabel("ROC-AUC")
    b.set_ylim(0.58, 0.75)
    b.yaxis.grid(True, color=GRID, lw=0.5)
    b.legend(loc="lower right", fontsize=6.3, borderaxespad=0.1)
    b.set_title("(b) 題數與 AUC")
    c = add_ax(fig, 10.75, 0.95, 3.25, 7.05)
    st = it.iloc[::-1]
    for k_, r in enumerate(st.itertuples()):
        c.plot([1, r.OR_每SD], [k_, k_], color="#c9c8c3", lw=1.0)
        c.plot(r.OR_每SD, k_, "o", color=MCOL["screener"], ms=3.8, mec="white", mew=0.6)
        up = r.OR_每SD >= 1
        c.text(1.03, k_, f"{r.OR_每SD:.3f}（{r.各折入選次數}／5）", va="center", transform=c.get_yaxis_transform(),
               ha="left", fontsize=6.3)
    c.axvline(1, color=SEC, lw=0.6)
    c.set_xscale("log")
    c.set_xlim(0.62, 1.6)
    c.set_xticks([0.7, 0.8, 1, 1.25, 1.5])
    c.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    c.xaxis.set_minor_locator(plt.NullLocator())
    c.set_yticks(range(len(st)), [f"{r.入選順序}. {r.題目}" for r in st.itertuples()], fontsize=6.6)
    c.set_ylim(-0.6, len(st) - 0.4)
    clean_y(c)
    c.set_xlabel("每 1 SD 的勝算比（對數尺度）")
    c.set_title("(c) 最終 10 題（括號＝入選折數）")
    save(fig, "fig_screener.png")


# ====================================================================== 圖：SHAP（彙總版）
def fig_shap():
    imp = rd("prediction/tables/shap_importance_y_inc.csv").sort_values("排名").head(15)
    dom = rd("prediction/tables/shap_domain_importance.csv")
    fig = new_fig(7.6)
    a = add_ax(fig, 3.85, 0.95, 5.6, 6.2)
    ys = np.arange(len(imp))[::-1]
    for yy, r in zip(ys, imp.itertuples()):
        col = RED if r.方向_spearman > 0.3 else (BLUE if r.方向_spearman < -0.3 else GREY)
        a.barh(yy, r.平均絕對SHAP_有作答者, height=0.62, color=col, zorder=2)
    tag = {"A": "（A 卷）", "B": "（B 卷）"}
    a.set_yticks(ys, [f"{r.排名}. {r.中文.replace('/', '／')}{tag.get(r.題本, '')}" for r in imp.itertuples()], fontsize=6.6)
    a.set_ylim(-0.6, len(imp) - 0.4)
    a.set_xlim(0, 0.37)
    clean_y(a)
    a.set_xlabel("平均 |SHAP|（A／B 卷題只計有作答者）")
    a.set_title("(a) HGB 最終模型的前 15 名特徵")
    hs = [Patch(fc=RED, label="作答值越高，預測風險越高（ρ > .3）"), Patch(fc=BLUE, label="作答值越高，預測風險越低（ρ < –.3）"),
          Patch(fc=GREY, label="無明確方向（|ρ| < .3）")]
    a.legend(handles=hs, loc="lower right", fontsize=6.2, borderaxespad=0.1, handlelength=1.0)
    b = add_ax(fig, 12.35, 0.95, 2.9, 6.2)
    dt = dom.iloc[::-1]
    yb = np.arange(len(dt))
    b.barh(yb, dt["平均絕對組合SHAP"], height=0.55, color=NAVY, zorder=2)
    for yy, v in zip(yb, dt["平均絕對組合SHAP"]):
        b.text(v + 0.01, yy, hu(v, 2), va="center", fontsize=6.4)
    b.set_yticks(yb, [f"{a_}\n（{n_}）" if n_ != "全體" else a_ for a_, n_ in zip(dt["領域"], dt["計算樣本"])], fontsize=6.4,
                 linespacing=1.05)
    b.set_xlim(0, 0.62)
    clean_y(b)
    b.set_xlabel("平均 |組合 SHAP|")
    b.set_title("(b) 六個領域")
    save(fig, "fig_shap.png")


# ====================================================================== 圖：晚睡與深夜螢幕的預測
def fig_night_screen():
    ob = rd("prediction/tables/observed_rate_by_bin.csv")
    ob = ob[ob["特徵"] == "bedtime"].reset_index(drop=True)
    sens = rd("prediction/tables/adjusted_or_sensitivity.csv").set_index(["模型", "特徵"])
    dd = rd("prediction/tables/drop_domain_auc.csv")
    dd = dd[dd["範圍"] == "領域樣本"]
    fig = new_fig(11.4)
    # (a)
    a = add_ax(fig, 1.05, 6.55, 6.1, 3.9)
    x = np.arange(len(ob))
    a.bar(x, ob["rate"] * 100, width=0.62, color=BLUE, zorder=2)
    a.errorbar(x, ob["rate"] * 100, yerr=[(ob["rate"] - ob["lo"]) * 100, (ob["hi"] - ob["rate"]) * 100], fmt="none",
               ecolor=INK, elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=3)
    for xx, r in zip(x, ob.itertuples()):
        a.text(xx, r.hi * 100 + 0.8, hu(r.rate * 100, 1), ha="center", va="bottom", fontsize=6.2)
    labs = ["22:00\n前", "22:00\n–29", "22:30\n–59", "23:00\n–29", "23:30\n–59", "0:00\n–29", "0:30\n以後"]
    a.set_xticks(x, [f"{l}\nn={n:,}" for l, n in zip(labs, ob["n"])], fontsize=6.2, linespacing=1.0)
    a.tick_params(axis="x", length=0)
    a.set_ylim(0, 50)
    a.set_ylabel("國八新發生憂鬱或自傷意念（%）")
    a.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    a.set_axisbelow(True)
    a.set_title("(a) 國七上床時間與一年後新發生率（未調整）")
    # (b)
    b = add_ax(fig, 10.35, 6.55, 4.0, 3.9)
    FOREST = [("bedtime", "上床時間（每 1 SD）"), ("awake_midnight", "午夜仍醒著"), ("late_screen", "23 點後以螢幕為主"),
              ("fomo", "FOMO（每 1 SD）"), ("addict", "網路成癮（每 1 SD）")]
    PM = [("M1", "P1", GREY, 0.24), ("M2", "P2", BLUE, 0.0), ("M3", "P3", PURPLE, -0.24)]
    ys = np.arange(len(FOREST))[::-1]
    for mk, lab, col, dy in PM:
        for (f, _), yy in zip(FOREST, ys):
            r = sens.loc[(mk, f)]
            dot(b, yy + dy, r.OR, r.CI下, r.CI上, col, "o", filled=r.p < .05, ms=3.0, lw=0.8)
    b.axvline(1, color=SEC, lw=0.6)
    b.set_xscale("log")
    b.set_xlim(0.85, 2.6)
    b.set_xticks([1, 1.5, 2, 2.5])
    b.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    b.xaxis.set_minor_locator(plt.NullLocator())
    b.set_yticks(ys, [l for _, l in FOREST], fontsize=6.6)
    b.set_ylim(-1.5, len(FOREST) - 0.4)
    clean_y(b)
    b.set_xlabel("勝算比（95% CI，對數尺度）")
    b.set_title("(b) 調整其他風險後的關聯")
    hs = [Line2D([], [], color=c, marker="o", ms=3.0, lw=0.8, label=l) for _, l, c, _ in PM]
    b.legend(handles=hs, loc="lower right", fontsize=6.3, borderaxespad=0.1, ncol=3, columnspacing=0.6, handlelength=1.2)
    # (c)
    c = add_ax(fig, 3.6, 1.0, 10.75, 4.2)
    groups = list(dict.fromkeys(dd["移除"]))
    yc = np.arange(len(groups))[::-1]
    for g, yy in zip(groups, yc):
        if g == "晚睡與深夜螢幕":
            c.axhspan(yy - 0.45, yy + 0.45, color="#fbe3d6", lw=0, zorder=0)
        for m, col, dy, fill in (("hgb", MCOL["hgb"], 0.15, True), ("lr", MCOL["lr"], -0.15, False)):
            r = dd[(dd["移除"] == g) & (dd["模型"] == m)].iloc[0]
            dot(c, yy + dy, r.ΔAUC, r.CI下, r.CI上, col, "o", filled=fill, ms=3.2, lw=0.9)
    smp = dd[dd["模型"] == "hgb"].set_index("移除")["評估樣本"]
    c.set_yticks(yc, [f"{g}{'（A 卷）' if smp[g] == 'A 卷作答者' else ('（B 卷）' if smp[g] == 'B 卷作答者' else '')}" for g in groups],
                 fontsize=6.6)
    c.set_ylim(-0.6, len(groups) - 0.4)
    c.axvline(0, color=SEC, lw=0.6)
    c.set_xlim(-0.012, 0.047)
    axfmt(c, "x", 2)
    clean_y(c)
    c.set_xlabel("ΔAUC＝全特徵 − 移除該組後（95% CI）")
    c.set_title("(c) 拿掉一組特徵，預測變差多少？")
    hs = [Line2D([], [], color=BLUE, marker="o", ms=3.2, lw=0.9, label="HGB"),
          Line2D([], [], color=MCOL["lr"], marker="o", ms=3.2, mfc="white", lw=0.9, label="邏輯斯迴歸")]
    c.legend(handles=hs, loc="lower right", fontsize=6.4, borderaxespad=0.1)
    save(fig, "fig_night_screen_prediction.png")


# ====================================================================== 圖：公平性
def fig_fairness():
    fa = rd("prediction/tables/fairness.csv")
    groups = list(dict.fromkeys(fa["子群"]))
    fig = new_fig(6.5)
    ys = np.arange(len(groups))[::-1]
    a = add_ax(fig, 2.85, 1.6, 5.6, 4.2)
    for m, col, mk, dy in (("hgb", MCOL["hgb"], "o", 0.15), ("screener", MCOL["screener"], "s", -0.15)):
        s = fa[fa["key"] == m].set_index("子群")
        for g, yy in zip(groups, ys):
            r = s.loc[g]
            dot(a, yy + dy, r.AUC, r.CI下, r.CI上, col, mk, ms=3.3, lw=0.9)
    s = fa[fa["key"] == "hgb"].set_index("子群")
    a.axvline(s.loc["全體", "AUC"], color=BLUE, lw=0.6, ls=":")
    a.set_yticks(ys, [f"{g}（n = {int(s.loc[g, 'n']):,}）" for g in groups], fontsize=6.6)
    a.set_ylim(-0.6, len(groups) - 0.4)
    a.set_xlim(0.45, 0.9)
    axfmt(a, "x", 2)
    clean_y(a)
    a.set_xlabel("ROC-AUC（95% CI，子群內學校 bootstrap）")
    a.set_title("(a) 各子群的排序準確度")
    hs = [Line2D([], [], color=BLUE, marker="o", ms=3.3, lw=0.9, label="HGB"),
          Line2D([], [], color=MCOL["screener"], marker="s", ms=3.3, lw=0.9, label="10 題簡短篩檢")]
    a.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2, fontsize=6.4, borderaxespad=0.1)
    b = add_ax(fig, 9.55, 1.6, 5.2, 4.2)
    for col_, fill, dy, lab in (("敏感度", True, 0.15, "全樣本統一前 15% 門檻"), ("敏感度_子群內", False, -0.15, "子群內各取前 15%")):
        for g, yy in zip(groups, ys):
            r = s.loc[g]
            dot(b, yy + dy, r[col_] * 100, r[col_ + "_lo"] * 100, r[col_ + "_hi"] * 100, BLUE, "o", filled=fill, ms=3.3, lw=0.9)
    for g, yy in zip(groups, ys):
        b.text(s.loc[g, "敏感度_hi"] * 100 + 1.5, yy + 0.15, f"標記 {hu(s.loc[g, '標記比例'] * 100, 1)}%", va="center",
               fontsize=6.2, color=SEC)
    b.set_yticks(ys, [])
    b.set_ylim(-0.6, len(groups) - 0.4)
    b.set_xlim(0, 95)
    clean_y(b)
    b.set_xlabel("HGB 敏感度：被標記到的新發生個案（%）")
    b.set_title("(b) 門檻設定與敏感度")
    hs = [Line2D([], [], color=BLUE, marker="o", ms=3.3, lw=0.9, label="全樣本統一前 15%"),
          Line2D([], [], color=BLUE, marker="o", ms=3.3, mfc="white", lw=0.9, label="子群內各取前 15%")]
    b.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2, fontsize=6.4, borderaxespad=0.1)
    save(fig, "fig_fairness.png")


# ====================================================================== 圖：休閒上網時數 vs 上床時間（腳本 09）
def fig_screen_vs_bedtime():
    """模型代號與表 tab:screen_hours 一致（①–④；不用晚間類型迴歸的 M1–M3）。"""
    co = rd("screen_vs_bedtime/tables/coefficients.csv")
    di = rd("screen_vs_bedtime/tables/difference_test.csv").set_index("設定")
    MAIN, CONV = "主樣本（全部 A 卷）", "課堂 A 卷 + 數位分區（= 03 慣例）"
    rows = [("M2 只放上網時數", "net_leisure_1", "① 只放休閒上網時數", RED, "o"),
            ("M3 只放上床時間", "bedtime_1", "只放上床時間", BLUE, "s"),
            ("M1 全模型", "net_leisure_1", "④ 全模型：休閒上網時數", RED, "o"),
            ("M1 全模型", "bedtime_1", "④ 全模型：上床時間", BLUE, "s"),
            (None, None, f"④ 差值：上床 {DASH} 上網", SEC, "D")]
    Wf = 15.0
    fig = new_fig(6.4, Wf)
    ax = add_ax(fig, 3.55, 1.55, Wf - 3.75, 4.7)
    ys = np.array([4.4, 3.4, 1.9, 0.9, -0.6])
    for setting, off, fill in ((MAIN, 0.17, True), (CONV, -0.17, False)):
        for (m, t, _, col, mk), y in zip(rows, ys + off):
            if m is None:
                d = di.loc[setting]
                est, lo, hi, p = d["差值"], d["CI下"], d["CI上"], d["p"]
            else:
                r = co[(co["設定"] == setting) & (co["模型"] == m) & (co["term"] == t)].iloc[0]
                est, lo, hi, p = r["係數"], r["CI下"], r["CI上"], r["p"]
            dot(ax, y, est, lo, hi, col, mk, filled=fill, ms=3.6, lw=1.3 if fill else 0.9)
            ax.text(hi + 0.025, y, f"{num(est, 3, plus=True)}（{pfmt(p)}）", va="center", fontsize=6.4, color=SEC)
    ax.axvline(0, color=SEC, lw=0.6, zorder=1)
    for yy in (2.65, 0.15):
        ax.axhline(yy, color=GRID, lw=0.7, zorder=0)
    ax.set_yticks(ys, [r[2] for r in rows], fontsize=7)
    ax.set_ylim(-1.05, 4.85)
    ax.set_xlim(-0.35, 1.45)
    ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    axfmt(ax, "x", 1)
    clean_y(ax)
    ax.set_xlabel("國七每多 1 SD 的國八憂鬱分數差（0–44 分；95% CI）")
    n_main = int(co[(co["設定"] == MAIN)]["n"].iloc[0])
    n_conv = int(co[(co["設定"] == CONV)]["n"].iloc[0])
    hs = [Line2D([], [], color=SEC, marker="o", ms=3.6, ls="none", label=f"實心：全部 A 卷 n = {n_main:,}"),
          Line2D([], [], color=SEC, marker="o", ms=3.6, mfc="white", ls="none",
                 label=f"空心：課堂 A 卷＋數位發展分區 n = {n_conv:,}")]
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, columnspacing=2.0, borderaxespad=0.15)
    save(fig, "fig_screen_vs_bedtime.png")


# ====================================================================== 圖：CLPM 睡眠時數版（版面同腳本 06 的上床時間版）
def fig_clpm_sleephrs():
    cl = rd("mediation/tables/clpm_paths.csv")
    cl = cl[cl["版本"] == "睡眠時數"].set_index(["自", "至"])
    WH, H = 8.4, 5.55                       # 與圖 fig_clpm_bedtime（腳本 06，8.4 cm）同尺寸；報告中兩圖皆以 12 cm 寬放入
    fig = new_fig(H, WH)
    bx = add_ax(fig, 0, 0, WH, H)
    bx.set_xlim(0, WH)
    bx.set_ylim(0, H)
    bx.axis("off")
    VARS = [("addict", "網路成癮"), ("sleep_hrs", "睡眠時數"), ("cesd_nosleep", "憂鬱症狀")]
    ys = {"addict": 4.12, "sleep_hrs": 2.72, "cesd_nosleep": 1.32}
    xl, xr, bw, bh = 0.88, WH - 0.88, 1.56, 0.56
    AR, NS = "#d6d4cd", "#a8a69f"
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

    def label(x, y, s_, color=INK, bold=False):
        bx.text(x, y, s_, ha="center", va="center", fontsize=7, color=color, fontweight="bold" if bold else "normal",
                zorder=5, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))

    segs, rects = {}, [(xc, ys[v], bw + 0.1, bh + 0.1) for xc in (xl, xr) for v, _ in VARS]
    for v, _ in VARS:
        segs[(v, v)] = ((xa, ys[v]), (xb, ys[v]))
        arrow(*segs[(v, v)], AR, 1.4, z=1)
    KEY = ("sleep_hrs", "cesd_nosleep")
    CROSS = [KEY, ("cesd_nosleep", "sleep_hrs"), ("addict", "cesd_nosleep"),
             ("cesd_nosleep", "addict"), ("addict", "sleep_hrs"), ("sleep_hrs", "addict")]
    for s_, t in CROSS:
        down = ys[t] < ys[s_]
        segs[(s_, t)] = ((xa, ys[s_] + (-off if down else off)), (xb, ys[t] + (off if down else -off)))

    def clearance(cx, cy, w, h, own):
        """標籤矩形與其他線段、方塊、已放標籤的最小距離（cm）。"""
        best = np.inf
        for k_, (p0, p1) in segs.items():
            if k_ == own:
                continue
            tt = np.linspace(0, 1, 160)
            px, py = p0[0] + tt * (p1[0] - p0[0]), p0[1] + tt * (p1[1] - p0[1])
            best = min(best, np.hypot(np.maximum(abs(px - cx) - w / 2, 0), np.maximum(abs(py - cy) - h / 2, 0)).min())
        for rx, ry, rw, rh in rects:
            best = min(best, np.hypot(max(abs(rx - cx) - (w + rw) / 2, 0), max(abs(ry - cy) - (h + rh) / 2, 0)))
        return best

    for v, _ in VARS:
        label(xa + 0.5, ys[v], hu(cl.loc[(f"{v}_1", f"{v}_2"), "beta"], 2), color=SEC)
        rects.append((xa + 0.5, ys[v], 0.5, 0.3))
    for s_, t in CROSS:
        r = cl.loc[(f"{s_}_1", f"{t}_2")]
        p0, p1 = segs[(s_, t)]
        key, rev, sig = (s_, t) == KEY, (s_, t) == ("cesd_nosleep", "sleep_hrs"), r.q < .05
        col = RED if key else (BLUE if sig else NS)
        arrow(p0, p1, col, 1.5 if key else (1.0 if sig else 0.8), "-" if sig else (0, (3, 2)))
        cand = [(clearance(p0[0] + u * (p1[0] - p0[0]), p0[1] + u * (p1[1] - p0[1]), 0.68, 0.3, (s_, t)), u)
                for u in np.linspace(0.14, 0.86, 37)]
        u = max(cand, key=lambda c: (round(c[0], 2), -abs(c[1] - 0.5)))[1]
        lx, ly = p0[0] + u * (p1[0] - p0[0]), p0[1] + u * (p1[1] - p0[1])
        label(lx, ly, num(r.beta, 3), color=col if sig else SEC, bold=key or rev)
        rects.append((lx, ly, 0.68, 0.3))
    hs = [Line2D([], [], color=RED, lw=1.5, label="睡眠時數→憂鬱"), Line2D([], [], color=BLUE, lw=1.0, label="其他 q < .05"),
          Line2D([], [], color=NS, lw=0.8, ls=(0, (3, 2)), label="未達顯著"), Line2D([], [], color=AR, lw=1.4, label="自我迴歸")]
    bx.legend(handles=hs, loc="center", bbox_to_anchor=(0.5, 0.33 / H), ncol=4, handlelength=1.5, fontsize=7,
              handletextpad=0.35, columnspacing=0.8, borderaxespad=0)
    save(fig, "fig_clpm_sleephrs.png")


# ====================================================================== 沿用原圖、只裁去圖內標題
def crop_title(src, name):
    """第一段深色列＝圖內標題；從標題下方第一段空白之後開始保留。"""
    im = Image.open(OUT / src).convert("RGB")
    a = np.asarray(im)
    dark = (a.min(axis=2) < 200).any(axis=1)
    i = int(np.argmax(dark))
    while i < len(dark) and dark[i]:
        i += 1
    top = max(i + 4, 0)
    im.crop((0, top, im.width, im.height)).save(FIG / name, dpi=im.info.get("dpi", (200, 200)))
    print("寫出", FIG / name, f"（裁去上方 {top} px 圖內標題）")


def fig_prediction_performance():
    crop_title("prediction/figures/fig2_performance.png", "fig_prediction_performance.png")


def fig_sequence_index():
    crop_title("typology/figures/fig_s2_sequence_index.png", "fig_sequence_index.png")


if __name__ == "__main__":
    fig_k_selection()
    fig_type_timeline()
    fig_type_profiles()
    fig_type_outcomes()
    fig_mediation("上床時間", "fig_mediation_bedtime.png")
    fig_mediation("睡眠時數", "fig_mediation_sleephrs.png")
    fig_indirect_forest()
    fig_collider_dag()
    fig_rules_vs_scolding()
    fig_parenting_crosslag()
    fig_restrict_adjustment()
    fig_interaction()
    fig_screener()
    fig_shap()
    fig_night_screen()
    fig_fairness()
    fig_screen_vs_bedtime()
    fig_clpm_sleephrs()
    fig_prediction_performance()
    fig_sequence_index()
