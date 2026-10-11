"""睡眠中介與兩波交叉延宕：國七上網型態／熬夜 → 國八睡眠 → 國八憂鬱？

1. 半縱貫中介（Cole & Maxwell, 2003）：X=國七暴露，M=國八上床時間（或睡眠時數），Y=國八憂鬱（排除睡眠題）
   a：M2 ~ X1 + M1 + Y1 + 共變項；b：Y2 ~ M2 + X1 + M1 + Y1 + 共變項（M2、Y2 同期測量）
   Cole–Maxwell 時序版：b 改用 M1 → Y2（控制 X1、Y1），各路徑皆有時間先後，但須假設穩定性。
   中介比例以「時序版（下界）–同期版（上界）」並列。間接效果 a×b：學校群集 bootstrap（B=2000）百分位 CI、BH-FDR。
   晚間日誌暴露分三類（23:00–02:00 已入睡／醒著做非螢幕活動／醒著用螢幕），以「螢幕 − 非螢幕」檢驗螢幕特異性。
2. 兩波 CLPM（網路成癮、上床時間／睡眠時數、憂鬱）：加權路徑迴歸＋學校群集 SE／bootstrap；方向比較用帶號差
   （睡眠時數取負以對齊預期方向）；semopy（ML，未加權）巢套模型，CFI/TLI 以 fixed-x 基準模型計算。兩波無法識別 RI-CLPM。
3. 敏感度：題本平衡（課堂 CA vs CB）、含課後補填 A 卷者、日誌螢幕特異性、排除國七高憂鬱、E-value（VanderWeele & Ding, 2017）。
4. 路徑圖、CLPM 圖、間接效果森林圖。

題本設計：國七 A/B 卷「以學校為單位」分派（同校各班同卷；國七原始 179 校中課堂 A 卷 89 校、B 卷 90 校），並非班內隨機；
另有少數 B 卷學生於課後（截止 9/15）自願補填 A 卷。國七睡眠／日誌／FOMO 只在 A 卷，故主分析 = 課堂 A 卷 89 校
（排除課後補填者：其「過去一個月」作息可能落在暑假，且為自選）；含補填者為敏感度分析。全樣本模型另控制課堂題本。
共變項：性別、家庭經濟（國七）、數位發展分區（學校層級）。

執行：.venv/Scripts/python -I scripts/03_sleep_mediation.py
輸出：outputs/mediation/{figures,tables}/、outputs/mediation/results.md
"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
import semopy
import statsmodels.api as sm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import SpecificationWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tigps import data as D  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
OUT = ROOT / "outputs" / "mediation"
FIG, TAB = OUT / "figures", OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

B, SEED = 2000, 20261019
C_FWD, C_REV, C_EXC, C_NS, C_AR = "#0072B2", "#D55E00", "#009E73", "#b5b5b5", "#555555"  # Okabe–Ito
C_RECIP = "#CC79A7"
LOG: list[str] = []
# GLM＋var_weights 的群集 SE 會發出警告；已與手算加權群集 sandwich 核對一致，故關閉此警告
warnings.filterwarnings("ignore", category=SpecificationWarning)


def log(s=""):
    print(s)
    LOG.append(s)


def fp(p):
    return "<.001" if p < .001 else f"{p:.3f}"


def fe(p):
    """內文用：p<.001 或 p=0.023（前面接 p 或 q）。"""
    return "<.001" if p < .001 else f"={p:.3f}"


def star(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def ci(lo, hi, k=3):
    return f"[{lo:.{k}f}, {hi:.{k}f}]"


def md_table(df):
    rows = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * df.shape[1]]
    return "\n".join(rows + ["| " + " | ".join(map(str, r)) + " |" for r in df.astype(str).values])


def wmean(x, w):
    return np.average(x, weights=w)


def wsd(x, w):
    return np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w))


def wls_coef(y, X, w):
    Xw = X * w[:, None]
    try:
        return np.linalg.solve(X.T @ Xw, Xw.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(X.T @ Xw, Xw.T @ y, rcond=None)[0]


def wls_fit(y, X, d):
    return sm.WLS(y, X, weights=d["HOUWGT"]).fit(cov_type="cluster", cov_kwds={"groups": d["school"]})


def boot_p(bs):
    """雙尾 bootstrap p：2×min(P(θ*≤0), P(θ*≥0))，下限 1/B。"""
    bs = bs[~np.isnan(bs)]
    return max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / len(bs))


def pct(bs):
    bs = np.asarray(bs, float)
    return np.nanpercentile(bs, [2.5, 97.5]) if np.isfinite(bs).any() else np.array([np.nan, np.nan])


def sig_ci(lo, hi):
    return lo > 0 or hi < 0


# ---------- 0. 資料與題本設計 ----------
p0 = D.load_panel()
p = p0[p0["HOUWGT"].notna()].copy()
resp = pyreadstat.read_sav(str(D.W1_SAV), usecols=["nstudent_id", "response"])[0].set_index("nstudent_id")["response"]
p["resp_1"] = p["nstudent_id"].map(resp)                  # 1 共A附B、2 共B附A、3 共A、4 共B、5 附A
p["ca_1"] = p["resp_1"].isin([1, 3]).astype(float)        # 課堂作答 A 卷（CA）
screen = [k for k, v in D.DIARY_GROUPS.items() if v.startswith("螢幕")]   # 14–19
late = p[["diary_23_1", "diary_00_1", "diary_01_1"]]
obs = late.notna().any(axis=1) & p["has_A_1"].astype(bool)
p["late_screen_1"] = late.isin(screen).any(axis=1).astype(float).where(obs)            # 23–02 任一時段主要活動為螢幕
p["late_awake_1"] = (late.notna() & (late != 8)).any(axis=1).astype(float).where(obs)  # 任一時段不是睡覺
p["late_other_1"] = p["late_awake_1"] - p["late_screen_1"]                             # 醒著但沒有螢幕時段
DIGI = {"數位發展潛力區": "d_pot", "數位發展起步區": "d_start", "數位發展萌動區": "d_sprout"}  # 參照：成熟區
for lvl, nm in DIGI.items():
    p[nm] = (p["digi_level"] == lvl).astype(float).where(p["digi_level"].notna())
COV = ["female_1", "fam_econ_1", *DIGI.values()]
p["school"] = pd.factorize(p["nschool_id"])[0]
# 國八高憂鬱：排除睡眠題 11 題之題均 ≥4/3（= 12 題 0–48 總分 ≥16 的等比換算；非此 12 題版本經驗證之臨床切點）
p["dep_hi_2"] = (p["cesd_nosleep_2"] / 11 >= 4 / 3 - 1e-9).astype(float).where(p["cesd_nosleep_2"].notna())
p["cesd16_2"] = (p["cesd_2"] >= 16).astype(float).where(p["cesd_2"].notna())
A = p[p["ca_1"] == 1]                          # 主分析：課堂作答 A 卷
A_ALL = p[p["has_A_1"].astype(bool)]           # 敏感度：含課後補填 A 卷者
SUPP = p[p["resp_1"].isin([2, 5])]
CB = p[p["resp_1"].isin([2, 4])]
N_SCH = p["school"].max() + 1
SCH_DIGI = p.groupby("school")["digi_level"].first().reindex(range(N_SCH)).to_numpy()


def draws(sch):
    """學校群集 bootstrap：在分析樣本的學校中、依數位發展分區分層重抽；回傳 B×學校 抽中次數（作為權重乘數）。"""
    rng = np.random.default_rng(SEED)
    present = np.unique(sch)
    strata = [present[SCH_DIGI[present] == lv] for lv in pd.unique(SCH_DIGI[present])]
    out = np.zeros((B, N_SCH))
    for r in range(B):
        for st in strata:
            np.add.at(out[r], rng.choice(st, len(st)), 1)
    return out


XS = {  # 鍵：(X, 同時放入的日誌虛擬變項, 標籤)；三類日誌以不同參照組參數化
    "addict": ("addict_1", [], "網路成癮"),
    "awake": ("late_awake_1", [], "23 點後仍醒著"),
    "scr": ("late_screen_1", ["late_other_1"], "　其中：螢幕 vs 已入睡"),
    "oth": ("late_other_1", ["late_screen_1"], "　其中：非螢幕 vs 已入睡"),
    "dif": ("late_screen_1", ["late_awake_1"], "　螢幕 − 非螢幕（皆醒著）"),
    "fomo": ("fomo_1", [], "FOMO"),
}
MLAB = {"bedtime": "上床時間", "sleep_hrs": "睡眠時數"}
log("# 睡眠中介與兩波 CLPM（scripts/03_sleep_mediation.py）\n")
cls_sh = p.groupby(["nschool_id", "school_class"])["ca_1"].mean()
sch_sh = p.groupby("nschool_id")["ca_1"].mean()
log("## 0. 題本設計與樣本\n")
mixed_c, mixed_s = (((s_ > 0) & (s_ < 1)).sum() for s_ in (cls_sh, sch_sh))
log(f"- **國七 A/B 卷以學校為單位分派，不是班內隨機**：串接樣本 {len(cls_sh)} 班中，課堂 CA 比例介於 0% 與 100% 之間的班級 {mixed_c} 個、"
    f"學校 {mixed_s} 所（{len(sch_sh)} 校）——即同校各班同卷（原始國七資料亦然：各班 CA 比例皆 ≤5% 或 ≥95%）。"
    "使用手冊僅載明「半數學生填答 CA，半數 CB」，未說明學校如何分派，故 A 卷子樣本不能宣稱為隨機子樣本或無偏。")
log(f"- 樣本流程：串接 n={len(p0):,} → 有權重 n={len(p):,}（{N_SCH} 校）→ **主分析：課堂 A 卷 n={len(A):,}（{A['nschool_id'].nunique()} 校）**；"
    f"課堂 B 卷 n={len(CB):,}（{CB['nschool_id'].nunique()} 校），其中 {len(SUPP)} 人（{SUPP['nschool_id'].nunique()} 校，每校中位數 "
    f"{SUPP.groupby('nschool_id').size().median():.0f} 人）課後自願補填 A 卷，主分析排除、納入敏感度分析。")
need = ["addict_1", "bedtime_1", "cesd_nosleep_1", "bedtime_2", "cesd_nosleep_2", *COV]
miss = A[need].isna().sum()
cc = A[need].notna().all(axis=1)
w_ = A["HOUWGT"]
flow_txt = "、".join(f"{v} {k}" for v, k in miss.items() if k)
cmp_cc = {v: (wmean(A.loc[cc & A[v].notna(), v], w_[cc & A[v].notna()]), wmean(A.loc[~cc & A[v].notna(), v], w_[~cc & A[v].notna()]))
          for v in ("cesd_nosleep_1", "addict_1")}
log(f"- 主模型（成癮→上床時間→憂鬱）完整資料：課堂 A 卷 {len(A):,} 人中遺漏 {flow_txt}；完整 n={cc.sum():,}（流失 {1 - cc.mean():.1%}）。"
    f"未完整者國七憂鬱 {cmp_cc['cesd_nosleep_1'][1]:.2f} vs 完整者 {cmp_cc['cesd_nosleep_1'][0]:.2f}、成癮 {cmp_cc['addict_1'][1]:.2f} vs "
    f"{cmp_cc['addict_1'][0]:.2f}（加權）——完整資料樣本症狀較輕、成癮較低，若有偏誤，估計較可能偏向低估。")
pd.DataFrame({"變項": miss.index, "遺漏（課堂 A 卷）": miss.values}).to_csv(TAB / "sample_flow_missing.csv", index=False, encoding="utf-8-sig")
lv = A.dropna(subset=["late_awake_1"])
gw = {k: wmean(lv[c], lv["HOUWGT"]) for k, c in (("scr", "late_screen_1"), ("oth", "late_other_1"), ("awake", "late_awake_1"))}
log(f"- 國七晚間日誌（23:00–02:00，課堂 A 卷 n={len(lv):,}，加權）：仍醒著 {gw['awake']:.1%}＝以螢幕為主要活動 {gw['scr']:.1%}＋"
    f"只有非螢幕活動（寫作業、家事、補習等）{gw['oth']:.1%}；其餘已入睡。")
ok = p["dep_hi_2"].notna()
log(f"- 國八高憂鬱（排除睡眠題 11 題題均 ≥4/3，等同 12 題 0–48 總分 ≥16 的換算）加權盛行率 = "
    f"{wmean(p.loc[ok, 'dep_hi_2'], p.loc[ok, 'HOUWGT']):.1%}；12 題總分 ≥16 = "
    f"{wmean(p.loc[p.cesd16_2.notna(), 'cesd16_2'], p.loc[p.cesd16_2.notna(), 'HOUWGT']):.1%}。"
    "註：16 分是 20 題 CES-D／CESD-R（0–60）的慣用切點，套在此 12 題 0–48 版本只是約定門檻，非經驗證之臨床切點。")
log(f"- bootstrap B={B}（學校群集、依數位發展分區分層、seed={SEED}）。\n")

# 題本平衡：課堂 CA vs CB（國八無題本之分，國八差異反映學校組成；國七差異另可能含題序／情境效果）
BAL = [("female_1", "女性"), ("fam_econ_1", "家庭經濟（1–4）"), ("d_pot", "數位發展潛力區"), ("d_start", "數位發展起步區"),
       ("d_sprout", "數位發展萌動區"), ("cesd_nosleep_1", "國七憂鬱（排除睡眠題）"), ("addict_1", "國七網路成癮"),
       ("who5_1", "國七 WHO-5 幸福感"), ("grade_rank_1", "國七成績排名（大=差）"), ("life_sat_1", "國七生活滿意"),
       ("cohesion_1", "國七家庭凝聚"), ("cesd_nosleep_2", "國八憂鬱（排除睡眠題）"), ("addict_2", "國八網路成癮"),
       ("bedtime_2", "國八上床時間"), ("sleep_hrs_2", "國八睡眠時數"), ("fomo_2", "國八 FOMO"), ("who5_2", "國八 WHO-5")]
pb = p[p["resp_1"].isin([1, 2, 3, 4])]
brow = []
for v, lab in BAL:
    d = pb[[v, "ca_1", "HOUWGT", "school"]].dropna()
    f = wls_fit(d[v], sm.add_constant(d[["ca_1"]]), d)
    g1, g0 = d[d["ca_1"] == 1], d[d["ca_1"] == 0]
    brow.append(dict(變項=lab, CA=wmean(g1[v], g1["HOUWGT"]), CB=wmean(g0[v], g0["HOUWGT"]), 差=f.params["ca_1"],
                     d=f.params["ca_1"] / wsd(d[v].to_numpy(float), d["HOUWGT"].to_numpy(float)), p=f.pvalues["ca_1"],
                     n_CA=len(g1), n_CB=len(g0)))
bal = pd.DataFrame(brow)
bal.round(4).to_csv(TAB / "booklet_balance.csv", index=False, encoding="utf-8-sig")
log("### 題本平衡：課堂 A 卷（CA）vs 課堂 B 卷（CB）學校\n")
log("加權平均，差＝CA−CB，d＝差／全體加權 SD，p 為學校群集 SE（描述性，未校正多重比較）。國八沒有題本之分，故國八的差異反映兩組學校本身的組成；"
    "國七差異若大於國八，另可能含題序／情境效果（CA 卷中 CESD 緊接在睡眠、日誌題組之後）。\n")
log(md_table(pd.DataFrame({"變項": bal["變項"], "CA": bal.CA.round(3), "CB": bal.CB.round(3), "d": bal.d.round(3),
                           "p": [fp(v) for v in bal.p], "n CA/CB": [f"{a}/{b}" for a, b in zip(bal.n_CA, bal.n_CB)]})))
bal_d = bal.set_index("變項")


# ---------- 1. 半縱貫中介 ----------
def mediation(df, key, m, m1=True, sample="A 卷", extra=()):
    """回傳 (結果列, bootstrap DataFrame)。標準化為完全標準化（加權 SD）。"""
    X1, ex, xlab = XS[key]
    M2, Y2, Y1 = f"{m}_2", "cesd_nosleep_2", "cesd_nosleep_1"
    M1 = f"{m}_1" if m1 else None
    base = ex + [Y1] + ([M1] if m1 else []) + COV + list(extra)
    d = df[[X1, M2, Y2] + base + ["HOUWGT", "school"]].dropna()
    Z = sm.add_constant(d[[X1] + base])
    Zb = sm.add_constant(d[[M2, X1] + base])
    ft, fa, fb = wls_fit(d[Y2], Z, d), wls_fit(d[M2], Z, d), wls_fit(d[Y2], Zb, d)
    xv, mv, yv, w0, sch = (d[c].to_numpy(float) for c in (X1, M2, Y2, "HOUWGT", "school"))
    m1v = d[M1].to_numpy(float) if m1 else None
    sch = sch.astype(int)
    Zt, Zbt = Z.to_numpy(float), Zb.to_numpy(float)
    i_m1 = list(Z.columns).index(M1) if m1 else None
    binary = set(np.unique(xv)) <= {0.0, 1.0}

    def calc(w):
        ct, ca, cb = wls_coef(yv, Zt, w), wls_coef(mv, Zt, w), wls_coef(yv, Zbt, w)
        a, c, b, cp = ca[1], ct[1], cb[1], cb[2]
        bcm = ct[i_m1] if m1 else np.nan
        sx, sm_, sy = wsd(xv, w), wsd(mv, w), wsd(yv, w)
        sm1 = wsd(m1v, w) if m1 else np.nan
        return dict(a=a, b=b, c=c, cp=cp, b_cm=bcm, ab=a * b, ab_cm=a * bcm, pm=a * b / c, pm_cm=a * bcm / c,
                    a_std=a * sx / sm_, b_std=b * sm_ / sy, c_std=c * sx / sy, cp_std=cp * sx / sy,
                    b_cm_std=bcm * sm1 / sy, ab_std=a * b * sx / sy, ab_cm_std=a * bcm * sx / sy,
                    c_d=c * (1 if binary else sx) / sy, sx=sx, sm=sm_, sy=sy)

    est = calc(w0)
    dr = draws(sch)
    bs = pd.DataFrame([calc(w0 * dr[r, sch]) for r in range(B)])
    row = dict(樣本=sample, key=key, X=xlab.strip("　"), M=MLAB[m], x=X1, m=m, n=len(d), n_school=len(np.unique(sch)), binary=binary)
    se = {"a": fa.bse[X1], "b": fb.bse[M2], "c": ft.bse[X1], "cp": fb.bse[X1], "b_cm": ft.bse[M1] if m1 else np.nan}
    pv = {"a": fa.pvalues[X1], "b": fb.pvalues[M2], "c": ft.pvalues[X1], "cp": fb.pvalues[X1],
          "b_cm": ft.pvalues[M1] if m1 else np.nan}
    for k in ("a", "b", "c", "cp", "b_cm"):
        row.update({k: est[k], f"{k}_se": se[k], f"{k}_p": pv[k], f"{k}_std": est[f"{k}_std"]})
        row[f"{k}_std_lo"], row[f"{k}_std_hi"] = pct(bs[f"{k}_std"])
    for k in ("a", "ab", "ab_cm", "ab_std", "ab_cm_std", "pm", "pm_cm", "c_d"):
        row[k] = est[k]
        row[f"{k}_lo"], row[f"{k}_hi"] = pct(bs[k]) if bs[k].notna().any() else (np.nan, np.nan)
    row["ab_p"], row["ab_cm_p"] = boot_p(bs["ab"].to_numpy()), boot_p(bs["ab_cm"].to_numpy()) if m1 else np.nan
    row.update(sd_x=est["sx"], sd_m=est["sm"], sd_y=est["sy"])
    return row, bs


KEYS = ["addict", "awake", "scr", "oth", "dif", "fomo"]


def run_set(dfA, dfF, keys=KEYS, full=True):
    """一組中介模型：A 卷各暴露（控制 M1）＋全樣本網路成癮（無 M1，控制課堂題本）。"""
    out = []
    for m in ("bedtime", "sleep_hrs"):
        out += [mediation(dfA, k, m, True, "A 卷")[0] for k in keys]
        if full:
            out.append(mediation(dfF, "addict", m, False, "全樣本", ("ca_1",))[0])
    return pd.DataFrame(out)


med = run_set(A, p)
fam = pd.concat([med["ab_p"], med["ab_cm_p"]]).dropna()
q = pd.Series(multipletests(fam, method="fdr_bh")[1], index=fam.index)
med["ab_q"] = q.iloc[:len(med)].to_numpy()
med["ab_cm_q"] = np.nan
med.loc[med["ab_cm_p"].notna(), "ab_cm_q"] = q.iloc[len(med):].to_numpy()
med.round(5).to_csv(TAB / "mediation_paths.csv", index=False, encoding="utf-8-sig")


def pm_txt(v, lo, hi, cp):
    return f"{v:.0%} [{lo:.0%}, {hi:.0%}]" if cp < .05 and not np.isnan(v) else "—"


log("\n## 1. 半縱貫中介：國七 X → 國八 M → 國八憂鬱（排除睡眠題）\n")
log("a：X1→M2｜M1,Y1,共變項；b：M2→Y2｜X1,M1,Y1（同期）；b_CM：M1→Y2｜X1,Y1（Cole–Maxwell 時序版，需穩定性假設）。"
    "A 卷＝課堂作答 A 卷（主分析）；全樣本另控制課堂題本。係數為完全標準化（加權 SD），間接效果 CI 為學校群集 bootstrap 百分位；"
    f"q = BH-FDR（{len(fam)} 個間接效果，含時序版）。日誌三類以兩個虛擬變項同時放入：「螢幕／非螢幕 vs 已入睡」與「螢幕 − 非螢幕」"
    "為同一模型的不同參照組。中介比例只在總效果 p<.05 時報告：**時序版（b_CM，下界）與同期版（b，上界；含同時測量的共同變異）並列**。\n")
tab1 = pd.DataFrame({
    "X": med["X"] + "（" + med["樣本"] + "）", "M": med["M"], "n（校）": [f"{a}（{b}）" for a, b in zip(med.n, med.n_school)],
    "a (β)": [f"{v:.3f}{star(pp)}" for v, pp in zip(med.a_std, med.a_p)],
    "b 同期 (β)": [f"{v:.3f}{star(pp)}" for v, pp in zip(med.b_std, med.b_p)],
    "c 總效果 (β)": [f"{v:.3f}{star(pp)}" for v, pp in zip(med.c_std, med.c_p)],
    "c' 直接 (β)": [f"{v:.3f}{star(pp)}" for v, pp in zip(med.cp_std, med.cp_p)],
    "a×b (β) [95% CI]": [f"{v:.4f} {ci(lo, hi, 4)}" for v, lo, hi in zip(med.ab_std, med.ab_std_lo, med.ab_std_hi)],
    "q": [fp(v) for v in med.ab_q],
    "a×b_CM (β) [CI]": ["—" if np.isnan(v) else f"{v:.4f} {ci(lo, hi, 4)}"
                        for v, lo, hi in zip(med.ab_cm_std, med.ab_cm_std_lo, med.ab_cm_std_hi)],
    "q_CM": ["—" if np.isnan(v) else fp(v) for v in med.ab_cm_q],
    "中介比例 時序版 [CI]": [pm_txt(v, lo, hi, cp) for v, lo, hi, cp in zip(med.pm_cm, med.pm_cm_lo, med.pm_cm_hi, med.c_p)],
    "中介比例 同期版 [CI]": [pm_txt(v, lo, hi, cp) for v, lo, hi, cp in zip(med.pm, med.pm_lo, med.pm_hi, med.c_p)],
})
log(md_table(tab1))
tab1.to_csv(TAB / "mediation_summary.csv", index=False, encoding="utf-8-sig")
log("\n原始單位解讀（同期設計，b 為每晚睡 1 小時的 CESD 分數）：")
for r in med.itertuples():
    k = 1 if r.binary else r.sd_x
    unit = "二元（有 vs 無）" if r.binary else f"每 +1 SD（{r.sd_x:.2f}）"
    mu = "分鐘上床延後" if r.m == "bedtime" else "分鐘睡眠"
    log(f"- {r.X}（{r.樣本}）→ {r.M}：{unit} → 國八 {r.a * k * 60:+.1f} {mu} [{r.a_lo * k * 60:+.1f}, {r.a_hi * k * 60:+.1f}]；"
        f"b = {r.b:+.2f} CESD 分/小時；總效果 {r.c * k:+.2f} CESD 分，其中經睡眠 {r.ab * k:+.3f} 分 [{r.ab_lo * k:+.3f}, {r.ab_hi * k:+.3f}]")


# ---------- 2. 兩波 CLPM ----------
SEM_N = {"addict": "成癮", "bedtime": "上床時間", "sleep_hrs": "睡眠時數", "cesd_nosleep": "憂鬱"}


def sem_fit(desc, d, ys, xs):
    """semopy 把固定的外生共變數也算入動差，df 需扣 k(k+1)/2；指標依校正後 df 重算。
    CFI/TLI 的基準模型採 fixed-x：國八變項彼此獨立且不受國七變項預測、外生變項共變數自由（df = p_y(p_y−1)/2 + p_y·p_x）。
    semopy 的 AIC/BIC 在巢套模型間不可比，改報 ΔAIC＝χ²−2df、ΔBIC＝χ²−df·ln(n)（相對飽和模型；負值＝偏好限制模型）。"""
    mdl = semopy.Model(desc)
    mdl.fit(d)
    st = semopy.calc_stats(mdl).T["Value"]
    n, py, px = len(d), len(ys), len(xs)
    chi2, df_ = float(st["chi2"]), int(round(st["DoF"] - px * (px + 1) / 2))
    if df_ == 0:
        chi2 = 0.0
    S = np.cov(d[ys + xs].to_numpy(float).T, bias=True)
    S0 = S.copy()
    S0[:py, :py] = np.diag(np.diag(S[:py, :py]))
    S0[:py, py:] = S0[py:, :py] = 0
    chib, dfb = n * (np.linalg.slogdet(S0)[1] - np.linalg.slogdet(S)[1]), py * (py - 1) // 2 + py * px  # semopy 同為 n·F_ML
    rmsea = np.sqrt(max(chi2 - df_, 0) / (df_ * (n - 1))) if df_ > 0 else 0.0
    cfi = 1 - max(chi2 - df_, 0) / max(chib - dfb, chi2 - df_, 1e-12)
    tli = ((chib / dfb) - (chi2 / df_)) / ((chib / dfb) - 1) if df_ > 0 else 1.0
    return mdl, dict(chi2=chi2, df=df_, p=stats.chi2.sf(chi2, df_) if df_ > 0 else np.nan, CFI=cfi, TLI=tli, RMSEA=rmsea,
                     dAIC=chi2 - 2 * df_, dBIC=chi2 - df_ * np.log(n), chi2_base=chib, df_base=dfb, n=n)


def clpm_beta(df, s):
    """交叉延宕路徑的加權標準化 β 與群集 p（無 bootstrap；敏感度用）。"""
    V = ["addict", s, "cesd_nosleep"]
    w1, w2 = [f"{v}_1" for v in V], [f"{v}_2" for v in V]
    d = df[w1 + w2 + COV + ["HOUWGT", "school"]].dropna()
    Z, w = sm.add_constant(d[w1 + COV]), d["HOUWGT"].to_numpy(float)
    out = []
    for y in w2:
        f = wls_fit(d[y], Z, d)
        out += [dict(自=x, 至=y, beta=f.params[x] * wsd(d[x].to_numpy(float), w) / wsd(d[y].to_numpy(float), w),
                     p=f.pvalues[x], n=len(d), n_school=d["school"].nunique()) for x in w1 if x[:-2] != y[:-2]]
    return pd.DataFrame(out)


def clpm(df, s):
    V = ["addict", s, "cesd_nosleep"]
    w1, w2 = [f"{v}_1" for v in V], [f"{v}_2" for v in V]
    d = df[w1 + w2 + COV + ["HOUWGT", "school"]].dropna()
    Z = sm.add_constant(d[w1 + COV])
    fits = {y: wls_fit(d[y], Z, d) for y in w2}
    Zt, w0, sch = Z.to_numpy(float), d["HOUWGT"].to_numpy(float), d["school"].to_numpy(int)
    Y = {c: d[c].to_numpy(float) for c in w1 + w2}
    pairs = [(x, y) for y in w2 for x in w1]

    def calc(w):
        out = {}
        for y in w2:
            cf, sy = wls_coef(Y[y], Zt, w), wsd(Y[y], w)
            for i, x in enumerate(w1):
                out[(x, y)] = cf[1 + i] * wsd(Y[x], w) / sy
        return out

    est = calc(w0)
    dr = draws(sch)
    bs = pd.DataFrame([calc(w0 * dr[r, sch]) for r in range(B)])
    tab = []
    for x, y in pairs:
        lo, hi = pct(bs[(x, y)])
        tab.append(dict(版本=SEM_N[s], 自=x, 至=y, 類型="自我迴歸" if x[:-2] == y[:-2] else "交叉延宕",
                        b=fits[y].params[x], se=fits[y].bse[x], p=fits[y].pvalues[x], beta=est[(x, y)],
                        beta_lo=lo, beta_hi=hi, n=len(d), n_school=len(np.unique(sch))))
    res = {y: (d[y] - fits[y].fittedvalues).to_numpy() for y in w2}
    rcor = {(a, b): np.average(res[a] * res[b], weights=w0) /
            np.sqrt(np.average(res[a] ** 2, weights=w0) * np.average(res[b] ** 2, weights=w0))
            for i, a in enumerate(w2) for b in w2[i + 1:]}
    ad, sl, dp = V
    # 帶號差：睡眠時數的預期方向為負，凡路徑一端為睡眠時數者乘 −1 以對齊（上床時間版不變）
    sg = lambda k: -1.0 if s == "sleep_hrs" and (k[0].startswith(sl) or k[1].startswith(sl)) else 1.0  # noqa: E731
    contr = {
        "成癮→憂鬱 vs 憂鬱→成癮": ((f"{ad}_1", f"{dp}_2"), (f"{dp}_1", f"{ad}_2")),
        f"成癮→{SEM_N[s]} vs {SEM_N[s]}→成癮": ((f"{ad}_1", f"{sl}_2"), (f"{sl}_1", f"{ad}_2")),
        f"{SEM_N[s]}→憂鬱 vs 憂鬱→{SEM_N[s]}": ((f"{sl}_1", f"{dp}_2"), (f"{dp}_1", f"{sl}_2")),
    }
    ctab = []
    for nm, (f, r) in contr.items():
        diff = sg(f) * bs[f] - sg(r) * bs[r]
        lo, hi = pct(diff)
        ctab.append(dict(版本=SEM_N[s], 比較=nm, 順向β=sg(f) * est[f], 反向β=sg(r) * est[r], 差=sg(f) * est[f] - sg(r) * est[r],
                         lo=lo, hi=hi, p_boot=boot_p(diff.to_numpy())))
    for nm, (k1, k2) in {f"鏈 成癮→{SEM_N[s]}→憂鬱": ((f"{ad}_1", f"{sl}_2"), (f"{sl}_1", f"{dp}_2")),
                         f"鏈 憂鬱→{SEM_N[s]}→成癮": ((f"{dp}_1", f"{sl}_2"), (f"{sl}_1", f"{ad}_2"))}.items():
        prod = bs[k1] * bs[k2]
        lo, hi = pct(prod)
        ctab.append(dict(版本=SEM_N[s], 比較=nm + "（β×β，穩定性假設）", 順向β=est[k1] * est[k2], 反向β=np.nan,
                         差=np.nan, lo=lo, hi=hi, p_boot=boot_p(prod.to_numpy())))
    # semopy：未加權 ML，巢套模型（拿掉某組交叉路徑）
    full_desc = lambda drop=(): "\n".join(  # noqa: E731
        [f"{y} ~ " + " + ".join([x for x in w1 if (x, y) not in drop] + COV) for y in w2]
        + [f"{a} ~~ {b}" for i, a in enumerate(w2) for b in w2[i + 1:]])
    models = {
        "M0 完整 CLPM（飽和）": (),
        "M1 無憂鬱→成癮/睡眠": ((f"{dp}_1", f"{ad}_2"), (f"{dp}_1", f"{sl}_2")),
        "M2 無成癮→睡眠/憂鬱": ((f"{ad}_1", f"{sl}_2"), (f"{ad}_1", f"{dp}_2")),
        "M3 無睡眠→成癮/憂鬱": ((f"{sl}_1", f"{ad}_2"), (f"{sl}_1", f"{dp}_2")),
        "M4 僅自我迴歸（無交叉路徑）": tuple((x, y) for x, y in pairs if x[:-2] != y[:-2]),
    }
    dsem = d[w1 + w2 + COV]
    ftab, sem_std = [], None
    for nm, drop in models.items():
        mdl, st = sem_fit(full_desc(drop), dsem, w2, w1 + COV)
        ftab.append(dict(版本=SEM_N[s], 模型=nm, **st))
        if not drop:
            ins = mdl.inspect(std_est=True)
            sem_std = ins[(ins.op == "~") & ins.rval.isin(w1)][["lval", "rval", "Estimate", "Est. Std", "p-value"]]
    return dict(paths=pd.DataFrame(tab), rcor=rcor, contr=pd.DataFrame(ctab), fit=pd.DataFrame(ftab),
                sem_std=sem_std.assign(版本=SEM_N[s]), n=len(d), n_school=len(np.unique(sch)), V=V)


CL = {s: clpm(A, s) for s in ("bedtime", "sleep_hrs")}
paths = pd.concat([c["paths"] for c in CL.values()], ignore_index=True)
cross = paths["類型"] == "交叉延宕"
paths["q"] = np.nan
paths.loc[cross, "q"] = multipletests(paths.loc[cross, "p"], method="fdr_bh")[1]
paths.round(5).to_csv(TAB / "clpm_paths.csv", index=False, encoding="utf-8-sig")
contr = pd.concat([c["contr"] for c in CL.values()], ignore_index=True)
contr.round(5).to_csv(TAB / "clpm_direction_contrasts.csv", index=False, encoding="utf-8-sig")
fit = pd.concat([c["fit"] for c in CL.values()], ignore_index=True)
fit.round(4).to_csv(TAB / "clpm_semopy_fit.csv", index=False, encoding="utf-8-sig")
pd.concat([c["sem_std"] for c in CL.values()]).round(4).to_csv(TAB / "clpm_semopy_std.csv", index=False, encoding="utf-8-sig")


def lab(v):
    return f"{SEM_N[v[:-2]]}{'₁' if v.endswith('_1') else '₂'}"


log("\n## 2. 兩波交叉延宕模型（CLPM，課堂 A 卷完整資料）\n")
log("加權 WLS 路徑迴歸（每個國八變項 ~ 三個國七變項 + 共變項），學校群集 SE；β = 加權完全標準化，CI = 群集 bootstrap；"
    "q = BH-FDR（兩版本共 12 條交叉路徑）。完整 CLPM 為飽和模型（df=0），故以 semopy ML（未加權）巢套模型比較檢定各組路徑；"
    "CFI/TLI 以 fixed-x 基準模型計算（國八變項彼此獨立、不受國七變項預測；外生變項共變數自由）。"
    "**兩波資料無法識別 RI-CLPM**（隨機截距需 ≥3 波），故交叉係數混合了穩定的個人間差異與個人內變化，不能解讀為個人內因果效果。\n")
for s, c in CL.items():
    pt = paths[paths["版本"] == SEM_N[s]]
    t = pd.DataFrame({"路徑": [f"{lab(a)} → {lab(b)}" for a, b in zip(pt["自"], pt["至"])], "類型": pt["類型"],
                      "β [95% CI]": [f"{v:.3f} {ci(lo, hi)}" for v, lo, hi in zip(pt.beta, pt.beta_lo, pt.beta_hi)],
                      "p": [fp(v) for v in pt.p], "q": ["" if np.isnan(v) else fp(v) for v in pt.q]})
    log(f"### 版本：{SEM_N[s]}（n={c['n']}，{c['n_school']} 校）\n")
    log(md_table(t))
    log("\n國八殘差相關：" + "；".join(f"{lab(a)}–{lab(b)} r={v:.3f}" for (a, b), v in c["rcor"].items()))
    ct = c["contr"]
    log("\n方向比較（帶號差 β順向 − β反向，群集 bootstrap；" + ("睡眠時數相關係數皆乘 −1 以對齊預期方向）：" if s == "sleep_hrs" else "預期方向皆為正）："))
    for r in ct.itertuples():
        if np.isnan(r.反向β):
            log(f"- {r.比較}：{r.順向β:.4f} {ci(r.lo, r.hi, 4)}，p{fe(r.p_boot)}")
        else:
            log(f"- {r.比較}：{r.順向β:.3f} vs {r.反向β:.3f}，差 {r.差:+.3f} {ci(r.lo, r.hi)}，p{fe(r.p_boot)}")
    ft = fit[fit["版本"] == SEM_N[s]]
    log(f"\nsemopy 巢套模型（未加權 ML；Δχ² 相對飽和模型 = 該模型 χ²；ΔBIC<0 表 BIC 偏好該限制模型；"
        f"基準模型 χ²={ft.chi2_base.iloc[0]:.1f}，df={ft.df_base.iloc[0]}）：\n")
    log(md_table(pd.DataFrame({"模型": ft["模型"], "χ²": ft.chi2.round(2), "df": ft.df, "p": [("—" if np.isnan(v) else fp(v)) for v in ft.p],
                               "CFI": ft.CFI.round(3), "TLI": ft.TLI.round(3), "RMSEA": ft.RMSEA.round(3),
                               "ΔAIC": ft.dAIC.round(1), "ΔBIC": ft.dBIC.round(1)})))
    log("")


# ---------- 3. 敏感度 ----------
def evalue(rr):
    rr = 1 / rr if rr < 1 else rr
    return rr + np.sqrt(rr * (rr - 1))


def evalue_ci(rr, lo, hi):
    if lo <= 1 <= hi:
        return 1.0
    return evalue(lo if rr > 1 else hi)


def lag_logit(df, x, controls, outcome):
    d = df[[outcome, x] + controls + ["HOUWGT", "school"]].dropna()
    xs = d[x] if set(d[x].unique()) <= {0, 1} else (d[x] - wmean(d[x], d.HOUWGT)) / wsd(d[x], d.HOUWGT)
    Z = sm.add_constant(pd.concat([xs.rename(x), d[controls]], axis=1))
    f = sm.GLM(d[outcome], Z, family=sm.families.Binomial(), var_weights=d["HOUWGT"]).fit(
        cov_type="cluster", cov_kwds={"groups": d["school"]})
    return f, d


def lag_linear(df, x, controls):
    d = df[["cesd_nosleep_2", x] + controls + ["HOUWGT", "school"]].dropna()
    f = wls_fit(d["cesd_nosleep_2"], sm.add_constant(d[[x] + controls]), d)
    binary = set(d[x].unique()) <= {0, 1}
    k = (1 if binary else wsd(d[x], d.HOUWGT)) / wsd(d["cesd_nosleep_2"], d.HOUWGT)
    lo, hi = f.conf_int().loc[x]
    return f.params[x] * k, lo * k, hi * k, f.pvalues[x], len(d)


Y1 = "cesd_nosleep_1"
ESPEC = [("網路成癮（A 卷）", A, "addict_1", [Y1, "bedtime_1"] + COV),
         ("網路成癮（全樣本，控制題本）", p, "addict_1", [Y1] + COV + ["ca_1"]),
         ("23 點後仍醒著（vs 已入睡）", A, "late_awake_1", [Y1, "bedtime_1"] + COV),
         ("23 點後仍醒著（不控制國七上床時間）", A, "late_awake_1", [Y1] + COV),
         ("FOMO（A 卷）", A, "fomo_1", [Y1, "bedtime_1"] + COV),
         ("上床時間（A 卷）", A, "bedtime_1", [Y1] + COV),
         ("睡眠時數（A 卷）", A, "sleep_hrs_1", [Y1] + COV)]
erows = []
for elab, df, x, ctrl in ESPEC:
    dd, dlo, dhi, dp_, nl = lag_linear(df, x, ctrl)
    rr_d, rr_lo, rr_hi = np.exp(.91 * dd), np.exp(.91 * dlo), np.exp(.91 * dhi)
    f, dl = lag_logit(df, x, ctrl, "dep_hi_2")
    or_, (olo, ohi) = np.exp(f.params[x]), np.exp(f.conf_int().loc[x])
    f16, _ = lag_logit(df, x, ctrl, "cesd16_2")
    rr_c, rlo, rhi = np.sqrt(or_), np.sqrt(olo), np.sqrt(ohi)
    erows.append(dict(暴露=elab, n=nl, d=dd, d_lo=dlo, d_hi=dhi, d_p=dp_, RR_d=rr_d, E_d=evalue(rr_d),
                      E_d_CI=evalue_ci(rr_d, rr_lo, rr_hi), OR=or_, OR_lo=olo, OR_hi=ohi, OR_p=f.pvalues[x],
                      OR_12item_ge16=np.exp(f16.params[x]), RR_sqrtOR=rr_c, E_OR=evalue(rr_c),
                      E_OR_CI=evalue_ci(rr_c, rlo, rhi), E_rare=evalue(or_), E_rare_CI=evalue_ci(or_, olo, ohi),
                      prev=wmean(dl["dep_hi_2"], dl["HOUWGT"]), n_logit=len(dl)))
ev = pd.DataFrame(erows)
ev.round(4).to_csv(TAB / "evalues.csv", index=False, encoding="utf-8-sig")
# 已測共變項的強度（同一 logistic 模型；連續者每 SD，取 ≥1 方向）作為未測混淆的對照
BN = {"cesd_nosleep_1": "國七憂鬱", "bedtime_1": "國七上床時間", "female_1": "女性", "fam_econ_1": "家庭經濟",
      "d_pot": "數位發展潛力區", "d_start": "數位發展起步區", "d_sprout": "數位發展萌動區"}
f_ref, d_ref = lag_logit(A, "addict_1", [Y1, "bedtime_1"] + COV, "dep_hi_2")
bench = {}
for c_ in [Y1, "bedtime_1"] + COV:
    k = 1 if set(d_ref[c_].unique()) <= {0, 1} else wsd(d_ref[c_].to_numpy(float), d_ref["HOUWGT"].to_numpy(float))
    o = np.exp(f_ref.params[c_] * k)
    bench[c_] = (o, max(o, 1 / o))
bench_s = sorted(((BN[k], v) for k, v in bench.items()), key=lambda t: -t[1][1])
top_nm, (top_or, top_inv) = next((n_, v) for n_, v in bench_s if n_ != "國七憂鬱")
e_ad = ev[ev["暴露"] == "網路成癮（A 卷）"].iloc[0]
log("## 3. 敏感度分析\n")
log("### 3a. E-value（VanderWeele & Ding, 2017）\n")
log(f"延宕關聯 = 國七暴露 → 國八憂鬱（排除睡眠題），控制國七憂鬱、國七上床時間（A 卷）、共變項。"
    "連續結果：d = 每 1 SD 暴露（二元暴露為有 vs 無）的國八憂鬱 SD 差，RR≈exp(0.91d)。"
    f"二元結果：國八高憂鬱（題均 ≥4/3），加權 logistic；結果加權盛行率 {e_ad.prev:.1%}（成癮 A 卷模型），略低於 15% 的常用門檻，"
    "依規則可視為罕見結果而以 OR≈RR（E_rare，較大）；此處以較保守的 √OR 轉換為主，兩者並列。\n")
log(md_table(pd.DataFrame({
    "暴露": ev["暴露"], "n": ev.n, "d [95% CI]": [f"{a:.3f} {ci(b, c)}" for a, b, c in zip(ev.d, ev.d_lo, ev.d_hi)],
    "E(d) / CI 界": [f"{a:.2f} / {b:.2f}" for a, b in zip(ev.E_d, ev.E_d_CI)],
    "OR [95% CI]": [f"{a:.2f} {ci(b, c, 2)}" for a, b, c in zip(ev.OR, ev.OR_lo, ev.OR_hi)],
    "E(√OR，保守) / CI 界": [f"{a:.2f} / {b:.2f}" for a, b in zip(ev.E_OR, ev.E_OR_CI)],
    "E(OR，罕見結果) / CI 界": [f"{a:.2f} / {b:.2f}" for a, b in zip(ev.E_rare, ev.E_rare_CI)],
    "OR（12 題總分 ≥16）": ev.OR_12item_ge16.round(2)})))
log("\n對照（同一 logistic 模型中已測共變項與國八高憂鬱的關聯，連續者每 SD，取 ≥1 方向）："
    + "、".join(f"{n_} {v[1]:.2f}" for n_, v in bench_s)
    + f"。除國七憂鬱外最強者為{top_nm}（OR {top_or:.2f}，相當於 {top_inv:.2f}；√OR {np.sqrt(top_inv):.2f}）。"
    f"成癮的 E 值 CI 界 {e_ad.E_OR_CI:.2f} {'低於' if e_ad.E_OR_CI < np.sqrt(top_inv) else '高於'}此 √OR：一個在國七憂鬱之外、"
    f"與暴露和結果的關聯都約如{top_nm}這般強的未測混淆，就足以讓 CI 涵蓋 1。故穩健性屬偏弱至中等（modest）。")

# 3b. 排除國七已高憂鬱者
p_lo = p[p["cesd_1"] < 16]
A_lo = p_lo[p_lo["ca_1"] == 1]
log(f"\n### 3b. 排除國七 12 題 CESD-R 總分 ≥16（約定門檻；排除 {int((p['cesd_1'] >= 16).sum())} 人；串接 n={len(p_lo)}，課堂 A 卷 n={len(A_lo)}）\n")
exc = run_set(A_lo, p_lo)
exc.round(5).to_csv(TAB / "sensitivity_exclude_highcesd.csv", index=False, encoding="utf-8-sig")


def cmp_table(m0, m1, l0="主", l1="排除"):
    return pd.DataFrame({
        "X": m0["X"] + "（" + m0["樣本"] + "）", "M": m0["M"], f"n {l0}/{l1}": [f"{a}/{b}" for a, b in zip(m0.n, m1.n)],
        f"c {l0}": [f"{v:.3f}{star(pp)}" for v, pp in zip(m0.c_std, m0.c_p)],
        f"c {l1}": [f"{v:.3f}{star(pp)}" for v, pp in zip(m1.c_std, m1.c_p)],
        f"a×b {l0} [CI]": [f"{v:.4f} {ci(lo, hi, 4)}" for v, lo, hi in zip(m0.ab_std, m0.ab_std_lo, m0.ab_std_hi)],
        f"a×b {l1} [CI]": [f"{v:.4f} {ci(lo, hi, 4)}{'' if sig_ci(lo, hi) else '（含 0）'}"
                           for v, lo, hi in zip(m1.ab_std, m1.ab_std_lo, m1.ab_std_hi)],
    })


cmp_ = cmp_table(med, exc)
log(md_table(cmp_))
cmp_.to_csv(TAB / "sensitivity_compare.csv", index=False, encoding="utf-8-sig")
flip = [(f"{a.X}（{a.樣本}）→{a.M}", a, b) for a, b in zip(med.itertuples(), exc.itertuples())]
lost = [nm for nm, a, b in flip if sig_ci(a.ab_std_lo, a.ab_std_hi) and not sig_ci(b.ab_std_lo, b.ab_std_hi)]
gain_c = [nm for nm, a, b in flip if a.c_p >= .05 and b.c_p < .05]
log("\n間接效果由顯著變為 CI 含 0：" + ("、".join(lost) if lost else "無")
    + "；總效果由不顯著變為 p<.05：" + ("、".join(gain_c) if gain_c else "無") + "。")

# 3c. 含課後補填 A 卷者（全部 A 卷）
log(f"\n### 3c. 題本敏感度：納入課後補填 A 卷者（全部 A 卷 n={len(A_ALL)}，{A_ALL['nschool_id'].nunique()} 校）\n")
kb = ["addict", "awake", "dif", "fomo"]
allA = run_set(A_ALL, p, kb, full=False)
allA.round(5).to_csv(TAB / "sensitivity_all_A_booklet.csv", index=False, encoding="utf-8-sig")
m_sub = med[med["樣本"].eq("A 卷") & med["key"].isin(kb)].reset_index(drop=True)
cmpA = cmp_table(m_sub, allA, "課堂 A", "全部 A")
log(md_table(cmpA))
cmpA.to_csv(TAB / "sensitivity_all_A_compare.csv", index=False, encoding="utf-8-sig")
clA = pd.concat([clpm_beta(A_ALL, s).assign(版本=SEM_N[s]) for s in ("bedtime", "sleep_hrs")], ignore_index=True)
clA = clA.merge(paths[["版本", "自", "至", "beta", "p"]], on=["版本", "自", "至"], suffixes=("_全部A", "_課堂A"))
clA.round(5).to_csv(TAB / "sensitivity_all_A_clpm.csv", index=False, encoding="utf-8-sig")
log("\nCLPM 交叉路徑（β，課堂 A → 全部 A）：" + "；".join(
    f"{r.版本}版 {lab(r.自)}→{lab(r.至)} {r.beta_課堂A:.3f}{star(r.p_課堂A)}→{r.beta_全部A:.3f}{star(r.p_全部A)}" for r in clA.itertuples()))
alx = pd.Series(mediation(A_ALL[A_ALL["cesd_1"] < 16], "addict", "bedtime")[0])   # 全部 A 卷＋排除國七高憂鬱（原分析的 3b 主列）
log(f"\n全部 A 卷且排除國七 12 題總分 ≥16（n={alx['n']}）：成癮→上床→憂鬱 a×b β={alx['ab_std']:.4f} "
    f"{ci(alx['ab_std_lo'], alx['ab_std_hi'], 4)}{'' if sig_ci(alx['ab_std_lo'], alx['ab_std_hi']) else '（含 0）'}。")


# 3d. 日誌螢幕特異性：螢幕 vs 非螢幕（原始單位，群集 SE），並加控國七起床時間以減少基線作息的殘餘混淆
#     樣本限國八憂鬱非遺漏，使「基本控制」列與中介模型的 a 路徑完全一致
def spec_table(df):
    rows = []
    for y, m1, unit, k in (("bedtime_2", "bedtime_1", "分鐘，正＝較晚", 60), ("sleep_hrs_2", "sleep_hrs_1", "分鐘", 60),
                           ("cesd_nosleep_2", "bedtime_1", "CESD 分", 1)):
        for clab, extra in (("基本控制", []), ("＋國七起床時間", ["waketime_1"])):
            ctrl = [m1, Y1] + extra + COV
            d = df[list(dict.fromkeys([y, "cesd_nosleep_2", "late_screen_1", "late_other_1"] + ctrl + ["HOUWGT", "school"]))].dropna()
            f = wls_fit(d[y], sm.add_constant(d[["late_screen_1", "late_other_1"] + ctrl]), d)
            t = f.t_test("late_screen_1 - late_other_1 = 0")
            cs, co = f.conf_int().loc["late_screen_1"] * k, f.conf_int().loc["late_other_1"] * k
            tl, th = np.ravel(t.conf_int())[:2] * k
            rows.append(dict(結果=f"國八{SEM_N.get(y[:-2], y)}", 單位=unit, 控制=clab, n=len(d),
                             螢幕=f.params["late_screen_1"] * k, 螢幕_lo=cs.iloc[0], 螢幕_hi=cs.iloc[1], 螢幕_p=f.pvalues["late_screen_1"],
                             非螢幕=f.params["late_other_1"] * k, 非螢幕_lo=co.iloc[0], 非螢幕_hi=co.iloc[1], 非螢幕_p=f.pvalues["late_other_1"],
                             差=float(np.ravel(t.effect)[0]) * k, 差_lo=tl, 差_hi=th, 差_p=float(np.ravel(t.pvalue)[0])))
    return pd.DataFrame(rows)


spec = spec_table(A)
spec.round(4).to_csv(TAB / "late_night_specificity.csv", index=False, encoding="utf-8-sig")
log("\n### 3d. 深夜活動的螢幕特異性（課堂 A 卷；參照＝23 點後已入睡；WLS＋學校群集 SE）\n")
log("控制國七同一睡眠變項（憂鬱模型控制國七上床時間）、國七憂鬱、共變項。日誌的「23 點後仍醒著」本身也是國七作息的第二個指標，"
    "控制自陳上床時間後的效果可能部分反映其測量誤差（基線作息的殘餘混淆），故另加控國七起床時間。\n")
log(md_table(pd.DataFrame({
    "結果": spec["結果"] + "（" + spec["單位"] + "）", "控制": spec["控制"], "n": spec.n,
    "螢幕 vs 已入睡": [f"{a:+.1f} {ci(b, c, 1)}{star(pp)}" for a, b, c, pp in zip(spec.螢幕, spec.螢幕_lo, spec.螢幕_hi, spec.螢幕_p)],
    "非螢幕 vs 已入睡": [f"{a:+.1f} {ci(b, c, 1)}{star(pp)}" for a, b, c, pp in zip(spec.非螢幕, spec.非螢幕_lo, spec.非螢幕_hi, spec.非螢幕_p)],
    "螢幕 − 非螢幕": [f"{a:+.1f} {ci(b, c, 1)}" for a, b, c in zip(spec.差, spec.差_lo, spec.差_hi)],
    "p（差）": [fp(v) for v in spec.差_p]})))


# ---------- 4. 圖 ----------
def edge_pt(c, toward, w, h):
    dx, dy = toward[0] - c[0], toward[1] - c[1]
    t = min(w / 2 / abs(dx) if dx else np.inf, h / 2 / abs(dy) if dy else np.inf)
    return c[0] + dx * t, c[1] + dy * t


def draw_box(ax, c, text, w, h, fc="#f3f6fb", ec="#2f3e5c", fs=10.5):
    ax.add_patch(FancyBboxPatch((c[0] - w / 2, c[1] - h / 2), w, h, boxstyle="round,pad=0,rounding_size=.02",
                                fc=fc, ec=ec, lw=1.3, zorder=3))
    ax.text(*c, text, ha="center", va="center", fontsize=fs, zorder=4, linespacing=1.35, color="#1d2433")


def draw_arrow(ax, c0, c1, w, h, color, lw, dashed=False, style="-|>", rad=0.0):
    a, b = edge_pt(c0, c1, w, h), edge_pt(c1, c0, w, h)
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=10 + 2.2 * lw, lw=lw, color=color,
                                 linestyle=(0, (4, 3)) if dashed else "-", shrinkA=1.5, shrinkB=1.5, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}"))
    return a, b


def lbl(ax, xy, s, color="#1d2433", ha="center", fs=9.5, weight="normal"):
    ax.text(*xy, s, ha=ha, va="center", fontsize=fs, color=color, weight=weight, zorder=5,
            bbox=dict(boxstyle="round,pad=.18", fc="white", ec="none", alpha=.92))


BOXX = {"addict": "國七\n網路成癮", "awake": "國七\n23 點後仍醒著", "fomo": "國七\nFOMO"}


def med_panel(ax, r, title, extra=None):
    ax.set_xlim(0, 1)
    ax.set_ylim(-.05, .92)
    ax.set_aspect("equal")
    ax.axis("off")
    PX, PM, PY, W, H = (.17, .36), (.5, .76), (.83, .36), .28, .15
    draw_box(ax, PX, BOXX[r.name], W, H)
    draw_box(ax, PM, "國八\n" + r["M"], W, H, fc="#e8f1f8")
    draw_box(ax, PY, "國八\n憂鬱症狀", W, H)
    for k, c0, c1, side in (("a", PX, PM, -1), ("b", PM, PY, 1)):
        sig = r[f"{k}_p"] < .05
        a, b = draw_arrow(ax, c0, c1, W, H, C_FWD if sig else C_NS, 1.2 + 9 * abs(r[f"{k}_std"]), dashed=not sig)
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        lbl(ax, (mx + side * .075, my + .045), f"{k} = {r[f'{k}_std']:.2f}{star(r[f'{k}_p'])}",
            ha="right" if side < 0 else "left", color=C_FWD if sig else "#666666")
    sig = r["cp_p"] < .05
    draw_arrow(ax, PX, PY, W, H, C_AR if sig else C_NS, 1.2 + 9 * abs(r["cp_std"]), dashed=not sig)
    lbl(ax, (.5, .225), f"c' ={r['cp_std']:.2f}{star(r['cp_p'])}　（總效果 c = {r['c_std']:.2f}{star(r['c_p'])}）", fs=9)
    ab_sig = r["ab_q"] < .05   # 以 FDR q 判定顯著
    ax.text(.5, .14, f"間接效果 a×b = {r['ab_std']:.3f}  95% CI [{r['ab_std_lo']:.3f}, {r['ab_std_hi']:.3f}]  q{fe(r['ab_q'])}",
            ha="center", va="center", fontsize=9.5, color=C_FWD if ab_sig else "#666666", weight="bold")
    pm = (f"　中介比例 ≈ {r['pm_cm']:.0%}（時序）–{r['pm']:.0%}（同期）"
          if (ab_sig and r["c_p"] < .05) else ("　總效果不顯著，不計中介比例" if ab_sig else ""))
    ax.text(.5, .07, f"n = {r['n']:,}（{r['n_school']} 校）{pm}", ha="center", va="center", fontsize=8.8, color="#555555")
    if extra:
        ax.text(.5, -.01, extra, ha="center", va="center", fontsize=8.5, color=C_REV, linespacing=1.4)
    ax.set_title(title, fontsize=11.5, pad=2, color="#1d2433")


def med_figure(m, fname):
    sub = med[(med.m == m) & (med["樣本"] == "A 卷")].set_index("key")
    dif = sub.loc["dif"]
    mu = "上床晚" if m == "bedtime" else "睡眠"
    extra = (f"螢幕 vs 非螢幕（皆醒著）：a 差 {dif.a * 60:+.1f} 分鐘{mu}（p{fe(dif.a_p)}），"
             f"a×b 差 {dif.ab_std:.3f} [{dif.ab_std_lo:.3f}, {dif.ab_std_hi:.3f}]")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.0))
    for ax, k in zip(axes, ["addict", "awake", "fomo"]):
        med_panel(ax, sub.loc[k], {"addict": "① 網路成癮", "awake": "② 熬夜：23 點後仍醒著（vs 已入睡）",
                                   "fomo": "③ FOMO 錯失恐懼"}[k], extra if k == "awake" else None)
    fig.suptitle(f"國七的上網型態與熬夜，是否透過國八「{MLAB[m]}」影響國八憂鬱？（完全標準化係數）",
                 fontsize=14, weight="bold", y=.99, color="#1d2433")
    fig.text(.5, .015, f"課堂 A 卷串接樣本（89 校）；HOUWGT 加權 WLS，控制國七{MLAB[m]}、國七憂鬱、性別、家庭經濟、數位發展分區；憂鬱排除睡眠題。"
             f"*p<.05 **p<.01 ***p<.001（學校群集 SE）；CI 為學校群集 bootstrap {B} 次。"
             "\n藍色實線＝a、b 路徑 p<.05；深灰實線＝直接效果 c' p<.05；灰色虛線＝未達顯著；間接效果藍字＝FDR q<.05。b 路徑為國八同期測量，方向無法由資料確立；"
             "中介比例下界用時序版 b（國七睡眠→國八憂鬱）。",
             ha="center", fontsize=8.4, color="#555555", linespacing=1.5)
    fig.subplots_adjust(left=.01, right=.99, top=.89, bottom=.1, wspace=.04)
    fig.savefig(FIG / fname, dpi=200)
    plt.close(fig)


def clpm_figure(s, fname):
    c = CL[s]
    pt = paths[paths["版本"] == SEM_N[s]].set_index(["自", "至"])
    ad, sl, dp = c["V"]
    rows_y = {ad: .78, sl: .5, dp: .22}
    names = {ad: "網路成癮", sl: MLAB[s], dp: "憂鬱症狀"}
    X1, X2, W, H = .15, .85, .2, .12
    fig, ax = plt.subplots(figsize=(9.2, 6.8))
    fig.subplots_adjust(left=.01, right=.99, top=.93, bottom=.14)
    ax.set_xlim(-.02, 1.2)
    ax.set_ylim(.02, 1.0)
    ax.axis("off")
    (px0, py0), (px1, py1) = ax.transData.transform([(0, 0), (1, 1)])
    k_xy = (py1 - py0) / (px1 - px0)  # arc3 的彎曲量以螢幕座標計，換回資料座標用
    for v, y in rows_y.items():
        draw_box(ax, (X1, y), names[v], W, H)
        draw_box(ax, (X2, y), names[v], W, H)
    ax.text(X1, .92, "國七（2023）", ha="center", fontsize=12, weight="bold", color="#1d2433")
    ax.text(X2, .92, "國八（2024）", ha="center", fontsize=12, weight="bold", color="#1d2433")
    colmap = {(ad, sl): C_FWD, (sl, dp): C_FWD, (ad, dp): C_FWD, (sl, ad): C_RECIP, (dp, ad): C_REV, (dp, sl): C_REV}
    # 標籤沿箭頭的位置：順向放左半、反向放右半，錯開避免重疊
    T = {(ad, sl): .35, (ad, dp): .2, (sl, dp): .15, (sl, ad): .65, (dp, ad): .8, (dp, sl): .85}
    for (x, y), r in pt.iterrows():
        vx, vy = x[:-2], y[:-2]
        c0, c1 = (X1, rows_y[vx]), (X2, rows_y[vy])
        if vx == vy:
            draw_arrow(ax, c0, c1, W, H, C_AR, 1 + 7 * abs(r.beta))
            lbl(ax, (.5 if vx != sl else .40, rows_y[vx]), f"{r.beta:.2f}{star(r.p)}", color=C_AR, fs=9)
            continue
        sig = r.q < .05
        col = colmap[(vx, vy)] if sig else C_NS
        a, b = draw_arrow(ax, c0, c1, W, H, col, 1 + 14 * abs(r.beta), dashed=not sig)
        t = T[(vx, vy)]
        lx, ly = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
        lbl(ax, (lx, ly), f"{r.beta:.2f}{star(r.q)}", color=col if sig else "#777777", fs=9.5,
            weight="bold" if sig else "normal")
    vs = [ad, sl, dp]
    for i, j, rad in ((0, 1, .5), (1, 2, .5), (0, 2, .62)):  # 國八殘差相關：弧線，標籤置於弧頂
        r_ = c["rcor"][(f"{vs[i]}_2", f"{vs[j]}_2")]
        y0, y1 = rows_y[vs[i]], rows_y[vs[j]]
        ax.add_patch(FancyArrowPatch((X2 + W / 2, y0), (X2 + W / 2, y1), arrowstyle="<|-|>", mutation_scale=8, lw=.9,
                                     color="#888888", connectionstyle=f"arc3,rad={-rad}", zorder=1, shrinkA=0, shrinkB=0))
        ax.text(X2 + W / 2 + rad * (y0 - y1) / 2 * k_xy, (y0 + y1) / 2, f"r={r_:.2f}", fontsize=8.5, color="#555555",
                ha="center", va="center", zorder=5, bbox=dict(boxstyle="round,pad=.15", fc="white", ec="none"))
    handles = [Line2D([], [], color=C_FWD, lw=2.5, label="順向：成癮→睡眠→憂鬱、成癮→憂鬱"),
               Line2D([], [], color=C_RECIP, lw=2.5, label="睡眠→成癮（與成癮→睡眠互相強化）"),
               Line2D([], [], color=C_REV, lw=2.5, label="反向：憂鬱→成癮、憂鬱→睡眠"),
               Line2D([], [], color=C_NS, lw=2, ls=(0, (4, 3)), label="未達顯著（FDR q≥.05）"),
               Line2D([], [], color=C_AR, lw=2.5, label="自我迴歸（穩定性）")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(.48, -.1), ncol=2, fontsize=8.8, frameon=False)
    ax.set_title(f"兩波交叉延宕模型：網路成癮、{MLAB[s]}與憂鬱（課堂 A 卷 n={c['n']:,}，{c['n_school']} 校）",
                 fontsize=13.5, weight="bold", color="#1d2433", pad=6)
    fig.text(.5, .012, "數字為完全標準化係數（HOUWGT 加權迴歸，控制性別、家庭經濟、數位發展分區）；交叉路徑顏色／星號依 BH-FDR q 值。\n"
             "兩波 CLPM 無法分離個人間與個人內效果（RI-CLPM 需 ≥3 波），係數代表「相對於同儕的變化」之預測關聯，非因果效果。",
             ha="center", fontsize=8.3, color="#555555", linespacing=1.5)
    fig.savefig(FIG / fname, dpi=200)
    plt.close(fig)


def forest_figure(fname):
    order = [("addict", "A 卷", "網路成癮（A 卷）"), ("addict", "全樣本", "網路成癮（全樣本†）"),
             ("awake", "A 卷", "23 點後仍醒著"), ("scr", "A 卷", "　其中：螢幕 vs 已入睡"),
             ("oth", "A 卷", "　其中：非螢幕 vs 已入睡"), ("dif", "A 卷", "　螢幕 − 非螢幕"), ("fomo", "A 卷", "FOMO")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), sharey=True, sharex=True)
    series = [("ab_std", med, C_FWD, "o", "主分析（同期 b）", -.24), ("ab_cm_std", med, C_REV, "s", "Cole–Maxwell 時序版", 0),
              ("ab_std", exc, C_EXC, "^", "排除國七高憂鬱（同期 b）", .24)]
    for ax, m in zip(axes, ("bedtime", "sleep_hrs")):
        ax.axhspan(2.5, 5.5, color="#f4f4f4", zorder=0)
        for i, (k, smp, _) in enumerate(order):
            for col_, src, color, mk, slab, off in series:
                r = src[(src.key == k) & (src["樣本"] == smp) & (src.m == m)].iloc[0]
                if np.isnan(r[col_]):
                    continue
                ax.errorbar(r[col_], i + off, xerr=[[r[col_] - r[f"{col_}_lo"]], [r[f"{col_}_hi"] - r[col_]]], fmt=mk,
                            color=color, ms=6.5, mec="white", mew=.8, elinewidth=1.7, capsize=0,
                            label=slab if (i == 0 and ax is axes[0]) else None)
        ax.axvline(0, color="#888888", lw=1, zorder=1)
        ax.set_title(f"中介變項：國八{MLAB[m]}", fontsize=11.5)
        ax.set_xlabel("標準化間接效果 a×b（95% bootstrap CI）")
        ax.grid(axis="x", color="#e6e6e6", lw=.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(order)), [o[2] for o in order])
    axes[0].invert_yaxis()
    fig.legend(loc="upper center", bbox_to_anchor=(.5, .935), ncol=3, frameon=False, fontsize=9.5)
    fig.suptitle("經由睡眠的間接效果：大多為正，但量很小；熬夜的效果未確認為螢幕特有", fontsize=13.5, weight="bold", y=.99)
    fig.text(.01, .012, "† 全樣本無法控制國七上床時間／睡眠時數（B 卷無此題），a 路徑因此偏高，亦無 Cole–Maxwell 版本；已控制課堂題本。"
             "灰底＝晚間日誌三類（參照：23 點後已入睡）。\n二元暴露亦以 SD 完全標準化以便比較。課堂 A 卷 89 校；學校群集 bootstrap 2000 次。",
             fontsize=8.3, color="#555555", linespacing=1.5)
    fig.subplots_adjust(left=.19, right=.98, top=.8, bottom=.17, wspace=.08)
    fig.savefig(FIG / fname, dpi=200)
    plt.close(fig)


med_figure("bedtime", "mediation_bedtime.png")
med_figure("sleep_hrs", "mediation_sleephrs.png")
clpm_figure("bedtime", "clpm_bedtime.png")
clpm_figure("sleep_hrs", "clpm_sleephrs.png")
forest_figure("indirect_forest.png")


# ---------- 5. 重點結論（數字皆由上方結果帶入） ----------
def mrow(k, m, smp="A 卷", src=None):
    src = med if src is None else src
    return src[(src.key == k) & (src.m == m) & (src["樣本"] == smp)].iloc[0]


def prow(s, a, b):
    return paths[(paths["版本"] == SEM_N[s]) & (paths["自"] == a) & (paths["至"] == b)].iloc[0]


def crow(s, key):
    return contr[(contr["版本"] == SEM_N[s]) & contr["比較"].str.startswith(key)].iloc[0]


def erow(elab):
    return ev[ev["暴露"] == elab].iloc[0]


def abt(r, k=4):
    return f"β={r.ab_std:.{k}f} {ci(r.ab_std_lo, r.ab_std_hi, k)}"


def pmt(r):
    return f"{r.pm_cm:.1%} [{r.pm_cm_lo:.1%}, {r.pm_cm_hi:.1%}]（時序版）至 {r.pm:.0%} [{r.pm_lo:.0%}, {r.pm_hi:.0%}]（同期版）"


ad_b, ad_s, aw_b, aw_s = mrow("addict", "bedtime"), mrow("addict", "sleep_hrs"), mrow("awake", "bedtime"), mrow("awake", "sleep_hrs")
sc_b, ot_b, df_b, df_s = mrow("scr", "bedtime"), mrow("oth", "bedtime"), mrow("dif", "bedtime"), mrow("dif", "sleep_hrs")
sc_s, ot_s = mrow("scr", "sleep_hrs"), mrow("oth", "sleep_hrs")
fo_b, fo_s, adf_b = mrow("fomo", "bedtime"), mrow("fomo", "sleep_hrs"), mrow("addict", "bedtime", "全樣本")
ad_bx, ad_sx, aw_bx = mrow("addict", "bedtime", src=exc), mrow("addict", "sleep_hrs", src=exc), mrow("awake", "bedtime", src=exc)
df_bx = mrow("dif", "bedtime", src=exc)
al_b = allA[(allA.key == "addict") & (allA.m == "bedtime")].iloc[0]
al_d = allA[(allA.key == "dif") & (allA.m == "bedtime")].iloc[0]
e_aw, e_aw0, e_bt = erow("23 點後仍醒著（vs 已入睡）"), erow("23 點後仍醒著（不控制國七上床時間）"), erow("上床時間（A 卷）")
e_af = erow("網路成癮（全樣本，控制題本）")
sp_btw = spec.set_index(["結果", "控制"]).loc[("國八上床時間", "＋國七起床時間")]
pa = {k: prow("bedtime", *v) for k, v in {
    "ad_bt": ("addict_1", "bedtime_2"), "bt_dp": ("bedtime_1", "cesd_nosleep_2"), "ad_dp": ("addict_1", "cesd_nosleep_2"),
    "dp_ad": ("cesd_nosleep_1", "addict_2"), "dp_bt": ("cesd_nosleep_1", "bedtime_2"), "bt_ad": ("bedtime_1", "addict_2")}.items()}
fb = fit[fit["版本"] == "上床時間"].set_index("模型")
fb_r, fb_a, fb_s, fb_4 = (fb.loc[[i for i in fb.index if i.startswith(k)][0]] for k in ("M1", "M2", "M3", "M4"))
c_ad, c_bt, c_ab = crow("bedtime", "成癮→憂鬱 vs"), crow("bedtime", "上床時間→憂鬱 vs"), crow("bedtime", "成癮→上床時間 vs")
ch_f, ch_r = crow("bedtime", "鏈 成癮"), crow("bedtime", "鏈 憂鬱")
c_sh = crow("sleep_hrs", "睡眠時數→憂鬱 vs")
main3 = med[(med["樣本"] == "A 卷") & med.key.isin(["addict", "awake", "fomo"])]
ab_sig = main3.loc[main3.ab_q < .05, "ab_std"]
b_beta = lambda r: f"β={r.beta:.3f} {ci(r.beta_lo, r.beta_hi)}, q{fe(r.q)}"  # noqa: E731
log("\n## 4. 重點結論（皆為關聯性推論；見限制）\n")
log(f"1. **成癮 → 晚睡 → 憂鬱：成立，但只佔一小部分。** 課堂 A 卷（n={ad_b.n:,}，{ad_b.n_school} 校）控制國七上床時間與國七憂鬱後，"
    f"國七網路成癮每高 1 SD，國八上床時間晚 {ad_b.a * ad_b.sd_x * 60:.1f} 分鐘（a β={ad_b.a_std:.3f}, p{fe(ad_b.a_p)}）；"
    f"國八上床時間每晚 1 小時，同期憂鬱高 {ad_b.b:.2f} 分（b β={ad_b.b_std:.3f}, p{fe(ad_b.b_p)}）。間接效果 {abt(ad_b)}（q{fe(ad_b.ab_q)}）；"
    f"時序版 a×b_CM β={ad_b.ab_cm_std:.4f} {ci(ad_b.ab_cm_std_lo, ad_b.ab_cm_std_hi, 4)}"
    f"（q{fe(ad_b.ab_cm_q)}）。中介比例約 {pmt(ad_b)}——同期 b 含同時測量的共同變異，宜視為上界；其餘為不經睡眠的直接路徑"
    f"（c' β={ad_b.cp_std:.3f}）。以睡眠時數為中介亦同（{abt(ad_s)}，q{fe(ad_s.ab_q)}；中介比例 {pmt(ad_s)}）。"
    f"全樣本版本（控制課堂題本，但無法控制國七上床時間）間接效果較大（β={adf_b.ab_std:.4f}，同期版 {adf_b.pm:.0%}），應視為高估。")
log(f"2. **熬夜（23 點後仍醒著）預測一年後更晚睡，但「螢幕特有」未獲確立。** 國七晚間日誌 23 點後仍醒著者（{gw['awake']:.1%}，其中約一半以螢幕為主要活動），"
    f"一年後上床時間比其國七上床時間所預期的再晚 {aw_b.a * 60:.1f} 分鐘 [{aw_b.a_lo * 60:.1f}, {aw_b.a_hi * 60:.1f}]（a β={aw_b.a_std:.3f}, p{fe(aw_b.a_p)}），"
    f"經上床時間的間接效果約 +{aw_b.ab:.2f} CESD 分 [{aw_b.ab_lo:.2f}, {aw_b.ab_hi:.2f}]（{abt(aw_b)}, q{fe(aw_b.ab_q)}）。"
    f"分開看（同一模型，參照＝已入睡；CI 為 bootstrap）：以螢幕為主 {sc_b.a * 60:+.1f} 分鐘 [{sc_b.a_lo * 60:.1f}, {sc_b.a_hi * 60:.1f}]、"
    f"只做非螢幕活動（多為寫作業、家事）{ot_b.a * 60:+.1f} 分鐘 [{ot_b.a_lo * 60:.1f}, {ot_b.a_hi * 60:.1f}]；螢幕 − 非螢幕 {df_b.a * 60:+.1f} 分鐘 "
    f"[{df_b.a_lo * 60:.1f}, {df_b.a_hi * 60:.1f}]（群集 p{fe(df_b.a_p)}），間接效果之差 {abt(df_b)}（q{fe(df_b.ab_q)}）。"
    + ("主樣本中此差勉強排除 0，但不穩健：" if sig_ci(df_b.ab_std_lo, df_b.ab_std_hi) else "此差不顯著；")
    + f"加控國七起床時間後 a 差 {sp_btw.差:+.1f} 分鐘（p{fe(sp_btw.差_p)}）；納入補填者 {abt(al_d)}；排除國七高憂鬱 {abt(df_bx)}；"
    f"以睡眠時數為中介，a 差 {df_s.a * 60:+.1f} 分鐘睡眠（p{fe(df_s.a_p)}），{abt(df_s)}。"
    f"故較穩妥的說法是「熬夜（螢幕是最常見的活動）預示日後更晚睡」：螢幕組點估計較大，但與非螢幕組的差異證據薄弱。"
    f"另外，控制國七上床時間後，熬夜對國八憂鬱的總效果不顯著（d={e_aw.d:.3f} {ci(e_aw.d_lo, e_aw.d_hi)}；不控制時 d={e_aw0.d:.3f} "
    f"{ci(e_aw0.d_lo, e_aw0.d_hi)}），故不報中介比例；日誌本身也是國七作息的第二個指標，a 路徑可能部分反映自陳上床時間的測量誤差。")
log(f"3. **FOMO 幾乎不走睡眠。** FOMO → 國八上床時間 a β={fo_b.a_std:.3f}（p{fe(fo_b.a_p)}），間接效果 {abt(fo_b)}"
    f"{'，CI 含 0' if not sig_ci(fo_b.ab_std_lo, fo_b.ab_std_hi) else ''}；以睡眠時數為中介時 {abt(fo_s)}（q{fe(fo_s.ab_q)}"
    f"{'，FDR 校正後不顯著，屬邊緣' if fo_s.ab_q >= .05 else ''}）。"
    f"FOMO 與日後憂鬱的關聯主要是直接路徑（c' β={fo_b.cp_std:.3f}, p{fe(fo_b.cp_p)}）。")
log(f"4. **方向：晚睡→憂鬱為單向；成癮與晚睡互相強化；成癮→憂鬱大於反向但差異未達顯著。** CLPM（上床時間版，n={CL['bedtime']['n']}）："
    f"成癮₁→上床₂ {b_beta(pa['ad_bt'])}；上床₁→憂鬱₂ {b_beta(pa['bt_dp'])}；成癮₁→憂鬱₂ {b_beta(pa['ad_dp'])}。"
    f"憂鬱₁→成癮₂ {b_beta(pa['dp_ad'])}；憂鬱₁→上床₂ {b_beta(pa['dp_bt'])}；上床₁→成癮₂ {b_beta(pa['bt_ad'])}。"
    f"帶號差：上床→憂鬱 − 憂鬱→上床 = {c_bt.差:+.3f} {ci(c_bt.lo, c_bt.hi)}（p{fe(c_bt.p_boot)}）；成癮→憂鬱 − 憂鬱→成癮 = "
    f"{c_ad.差:+.3f} {ci(c_ad.lo, c_ad.hi)}（p{fe(c_ad.p_boot)}）；成癮→上床 − 上床→成癮 = {c_ab.差:+.3f} {ci(c_ab.lo, c_ab.hi)}（p{fe(c_ab.p_boot)}）。"
    f"跨波鏈 成癮→上床→憂鬱 β×β={ch_f.順向β:.4f} {ci(ch_f.lo, ch_f.hi, 4)}，反向鏈 憂鬱→上床→成癮 {ch_r.順向β:.4f} {ci(ch_r.lo, ch_r.hi, 4)}。"
    f"semopy 巢套比較：拿掉憂鬱→成癮/睡眠 Δχ²(2)={fb_r.chi2:.1f}, p{fe(fb_r.p)}, ΔBIC={fb_r.dBIC:+.1f}"
    f"{'（BIC 偏好拿掉）' if fb_r.dBIC < 0 else ''}；拿掉成癮→睡眠/憂鬱 Δχ²(2)={fb_a.chi2:.1f}, ΔBIC={fb_a.dBIC:+.1f}；"
    f"拿掉睡眠→成癮/憂鬱 Δχ²(2)={fb_s.chi2:.1f}, ΔBIC={fb_s.dBIC:+.1f}；僅自我迴歸 Δχ²({fb_4.df})={fb_4.chi2:.1f}, CFI={fb_4.CFI:.3f}, "
    f"TLI={fb_4.TLI:.3f}, RMSEA={fb_4.RMSEA:.3f}。結論：順向（成癮→睡眠→憂鬱）證據較強，但「憂鬱→成癮」反向路徑小而非零，不能完全排除。"
    f"「憂鬱₁ 不預測上床₂」也間接支持中介模型中 b 路徑以「晚睡→憂鬱」方向為主。睡眠時數版方向一致但較弱"
    f"（睡眠→憂鬱 − 反向 = {c_sh.差:+.3f} {ci(c_sh.lo, c_sh.hi)}，p{fe(c_sh.p_boot)}）。")
bal_w1, bal_w2 = bal_d.loc["國七憂鬱（排除睡眠題）"], bal_d.loc["國八憂鬱（排除睡眠題）"]
bal_a1, bal_a2 = bal_d.loc["國七網路成癮"], bal_d.loc["國八網路成癮"]
log(f"5. **穩健性。** (a) 題本：A 卷學校國七憂鬱、成癮較高（d={bal_w1.d:.2f}, p{fe(bal_w1.p)}；d={bal_a1.d:.2f}, p{fe(bal_a1.p)}），"
    f"國八差距較小（d={bal_w2.d:.2f}, p{fe(bal_w2.p)}；d={bal_a2.d:.2f}, p{fe(bal_a2.p)}），顯示兩組學校略有不同、國七另可能有題序效果；"
    f"納入課後補填 A 卷者，成癮→上床→憂鬱 {ad_b.ab_std:.4f}→{abt(al_b)}"
    f"{'（仍顯著）' if sig_ci(al_b.ab_std_lo, al_b.ab_std_hi) else '（CI 含 0）'}，CLPM 交叉路徑幾乎不變；但螢幕 − 非螢幕的間接效果差 "
    f"{df_b.ab_std:.4f}→{abt(al_d)}{'' if sig_ci(al_d.ab_std_lo, al_d.ab_std_hi) else '（CI 含 0）'}，螢幕特異性對樣本定義敏感。"
    f"(b) E-value：國七網路成癮→國八高憂鬱 OR={e_ad.OR:.2f} {ci(e_ad.OR_lo, e_ad.OR_hi, 2)}（盛行率 {e_ad.prev:.1%}），"
    f"保守 √OR 轉換 E={e_ad.E_OR:.2f}（CI 界 {e_ad.E_OR_CI:.2f}），罕見結果 OR≈RR 轉換 E={e_ad.E_rare:.2f}（CI 界 {e_ad.E_rare_CI:.2f}）；"
    f"全樣本 E={e_af.E_OR:.2f}（CI 界 {e_af.E_OR_CI:.2f}）；國七上床時間 E={e_bt.E_OR:.2f}（CI 界 {e_bt.E_OR_CI:.2f}）。"
    f"已測共變項中除國七憂鬱外最強者為{top_nm}（相當 OR {top_inv:.2f}，√OR {np.sqrt(top_inv):.2f}），"
    f"一個這般強、且獨立於國七憂鬱的未測混淆就可能使 CI 涵蓋 1，故屬偏弱至中等穩健（modest）。"
    f"(c) 排除國七 12 題總分 ≥16 者：點估計大致穩定，多數總效果略增；成癮→上床→憂鬱 {ad_b.ab_std:.4f}→{abt(ad_bx)}"
    f"{'，CI 含 0（邊緣）' if not sig_ci(ad_bx.ab_std_lo, ad_bx.ab_std_hi) else '，仍顯著'}"
    + (f"（但若同時納入補填者，{abt(alx)}，CI 觸及 0，故此路徑屬邊緣穩健）" if not sig_ci(alx.ab_std_lo, alx.ab_std_hi) else "")
    + f"；睡眠時數版 {abt(ad_sx)}{'，仍顯著' if sig_ci(ad_sx.ab_std_lo, ad_sx.ab_std_hi) else '，CI 含 0'}；"
    f"熬夜的總效果 c β {aw_b.c_std:.3f}{star(aw_b.c_p)}→"
    f"{aw_bx.c_std:.3f}{star(aw_bx.c_p)}" + ("（排除後達 p<.05，故主樣本「總效果不顯著」並不穩健）" if aw_b.c_p >= .05 and aw_bx.c_p < .05 else "") + "。")
log(f"6. **限制。** (i) 只有兩波：中介的 b 路徑為國八同期測量（主分析），時序版需假設 M→Y 效果跨年穩定；"
    "CLPM 無法識別 RI-CLPM，交叉係數混合個人間與個人內差異。(ii) 國七睡眠／日誌／FOMO 只在 A 卷，而 A/B 卷以學校為單位分派、"
    f"分派方式未載明，A 卷分析只代表 {A['nschool_id'].nunique()} 所學校（有效群集數），且這些學校基線憂鬱與成癮略高；"
    "已以題本平衡表、含補填者的敏感度分析與全樣本題本控制處理，但無法排除學校層級差異。"
    f"(iii) 完整資料分析（未做多重插補；主模型流失 {1 - cc.mean():.0%}，流失者症狀較重）。(iv) 上床時間、日誌皆自陳；日誌為單日典型作息的粗略指標，"
    f"且晚間活動未區分螢幕上的內容。(v) 效果量小：A 卷顯著的完全標準化間接效果僅 {ab_sig.min():.3f}–{ab_sig.max():.3f} SD。")
log("\n圖：figures/mediation_bedtime.png、mediation_sleephrs.png、clpm_bedtime.png、clpm_sleephrs.png、indirect_forest.png")
(OUT / "results.md").write_text("\n".join(LOG) + "\n", encoding="utf-8")
