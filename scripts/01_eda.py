"""初步探索：樣本串接、量表信度、兩波變化、晚間作息熱力圖、跨波預測。

執行：.venv/Scripts/python scripts/01_eda.py
輸出：outputs/figures/*.png、outputs/tables/*.csv
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tigps import data as D  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)


def wmean(x, w):
    m = x.notna() & w.notna()
    return np.average(x[m], weights=w[m]) if m.any() else np.nan


def cronbach(df):
    df = df.dropna()
    k = df.shape[1]
    return k / (k - 1) * (1 - df.var().sum() / df.sum(axis=1).var())


# ---------- 1. 樣本 ----------
raw1, _ = D.read_wave(D.W1_SAV)
raw2, _ = D.read_wave(D.W2_SAV)
w1, w2 = D.build_w1(raw1), D.build_w2(raw2)
panel = D.load_panel()
print("== 樣本 ==")
print(f"W1 n={len(w1)}, W2 n={len(w2)}, 兩波串接 n={len(panel)} (W1 保留率 {len(panel)/len(w1):.1%})")
print("W1 題本:", w1["booklet"].value_counts().to_dict(),
      f"| 有 A 卷(作息/睡眠/上網時間) {w1.has_A.sum()}, 有 B 卷(數位教養) {w1.has_B.sum()}")

# ---------- 2. 信度 ----------
rel = pd.DataFrame({
    "量表": ["CESD-R 憂鬱(12題)", "網路成癮(10題)", "WHO-5 幸福感", "FOMO", "網路向上比較"],
    "W1 α": [cronbach(raw1[c]) for c in (D.CESD_W1, D.ADDICT_W1, D.WHO5_W1, D.FOMO_W1, D.SOCCOMP_W1)],
    "W2 α": [cronbach(raw2[c]) for c in (D.CESD_W2, D.ADDICT_W2, D.WHO5_W2, D.FOMO_W2, D.SOCCOMP_W2)],
}).round(3)
print("\n== 量表信度 ==\n", rel.to_string(index=False))
rel.to_csv(TAB / "reliability.csv", index=False, encoding="utf-8-sig")

# ---------- 3. 兩波加權平均變化（串接樣本） ----------
vars_ = {"cesd": "憂鬱 CESD (0-48)", "addict": "網路成癮 (1-4)", "who5": "WHO-5 (0-20)",
         "net_leisure": "平日休閒上網時數", "net_game": "平日遊戲時數", "net_chat": "平日聊天時數",
         "bedtime": "上課日上床時間", "sleep_hrs": "上課日睡眠時數", "life_sat": "生活滿意度 (1-4)",
         "cyber_victim": "網路霸凌受害比例", "bully_victim": "實體霸凌受害比例"}
w = panel["HOUWGT"]
chg = pd.DataFrame({
    "變項": list(vars_.values()),
    "國七": [wmean(panel[f"{v}_1"], w) for v in vars_],
    "國八": [wmean(panel[f"{v}_2"], w) for v in vars_],
}).round(2)
print("\n== 兩波變化（加權，串接樣本；國七睡眠/上網僅 A 卷）==\n", chg.to_string(index=False))
chg.to_csv(TAB / "wave_change.csv", index=False, encoding="utf-8-sig")
cut = 16  # CESD 常用臨界值
for wv, lab in [("1", "國七"), ("2", "國八")]:
    print(f"{lab} CESD≥{cut} 比例: {wmean((panel[f'cesd_{wv}'] >= cut).astype(float).where(panel[f'cesd_{wv}'].notna()), w):.1%}")
print(f"國八 近兩週有自傷/求死意念比例: {wmean(panel['selfharm_2'], w):.1%}")
late = panel["bedtime_2"] >= 24
print(f"國八 上課日午夜(含)後才睡比例: {wmean(late.astype(float).where(panel['bedtime_2'].notna()), w):.1%}"
      f"（國七 A 卷: {wmean((panel['bedtime_1']>=24).astype(float).where(panel['bedtime_1'].notna()), w):.1%}）")

# ---------- 4. 晚間作息熱力圖（W1, A 卷） ----------
slots = list(D.DIARY_SLOTS)
labels = ["18", "19", "20", "21", "22", "23", "0", "1"]
order = ["補習/課輔", "寫作業讀書", "運動社團才藝", "吃飯家事生活", "與家人朋友聊天",
         "螢幕：影音", "螢幕：遊戲/電視", "螢幕：社群聊天", "睡覺", "其他"]
A = w1[w1.has_A]
mat = pd.DataFrame(index=order, columns=labels, dtype=float)
for hr, lab in zip(slots, labels):
    g = A[f"diary_{hr}"].map(D.DIARY_GROUPS)
    ok = g.notna()
    share = A.loc[ok, "HOUWGT"].groupby(g[ok]).sum() / A.loc[ok, "HOUWGT"].sum()
    mat[lab] = share.reindex(order).fillna(0)
mat.to_csv(TAB / "evening_diary_share.csv", encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(8, 4.6))
im = ax.imshow(mat.values * 100, cmap="YlGnBu", aspect="auto", vmin=0)
ax.set_xticks(range(len(labels)), [f"{l}:00" for l in labels])
ax.set_yticks(range(len(order)), order)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat.iat[i, j] * 100
        if v >= 3:
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 40 else "black")
ax.set_title("國七學生的晚上在做什麼？（各時段主要活動 %，加權）")
fig.colorbar(im, ax=ax, label="%")
fig.tight_layout()
fig.savefig(FIG / "evening_heatmap.png", dpi=200)
screen = {"螢幕：影音", "螢幕：遊戲/電視", "螢幕：社群聊天"}
late_screen = A[["diary_23", "diary_00", "diary_01"]].apply(lambda s: s.map(D.DIARY_GROUPS).isin(screen)).any(axis=1)
print(f"\n國七 A 卷: 23:00 後任一時段以螢幕為主要活動的比例: {wmean(late_screen.astype(float), A['HOUWGT']):.1%}")

# ---------- 5. 上床時間分布 ----------
fig, ax = plt.subplots(figsize=(7, 3.6))
bins = np.arange(20, 29.01, .5)
for col, lab, c in [("bedtime_1", "國七 (2023)", "#9aa9c9"), ("bedtime_2", "國八 (2024)", "#2c4a8c")]:
    ok = panel[col].notna() & panel["HOUWGT"].notna()
    ax.hist(panel.loc[ok, col], bins=bins, weights=panel.loc[ok, "HOUWGT"],
            density=True, alpha=.65, label=lab, color=c)
ax.set_xticks(range(20, 30), [f"{h%24}:00" for h in range(20, 30)])
ax.set_xlabel("上課日上床時間"); ax.set_ylabel("密度"); ax.legend()
ax.set_title("同一批學生，升上國八後睡得更晚")
fig.tight_layout(); fig.savefig(FIG / "bedtime_shift.png", dpi=200)

# ---------- 6. 跨波預測：國七 → 國八憂鬱（控制國七憂鬱） ----------
def lagged(formula, df, label):
    cols = [c for c in df.columns if c in formula] + ["HOUWGT", "nschool_id"]
    d = df[list(dict.fromkeys(cols))].dropna()
    fit = smf.wls(formula, d, weights=d["HOUWGT"]).fit(cov_type="cluster", cov_kwds={"groups": d["nschool_id"]})
    print(f"\n-- {label} (n={int(fit.nobs)}) --")
    print(fit.summary2().tables[1][["Coef.", "Std.Err.", "P>|z|"]].round(3).to_string())
    return fit

print("\n== 跨波延宕迴歸：結果 = 國八 CESD（排除睡眠題），控制國七 CESD ==")
std = panel.copy()
for c in ["addict_1", "net_leisure_1", "bedtime_1", "fomo_1", "soccomp_1", "sleep_hrs_1"]:
    std[c] = (std[c] - std[c].mean()) / std[c].std()
lagged("cesd_nosleep_2 ~ cesd_nosleep_1 + addict_1 + female_1 + fam_econ_1", std, "全樣本：網路成癮")
lagged("cesd_nosleep_2 ~ cesd_nosleep_1 + addict_1 + net_leisure_1 + bedtime_1 + fomo_1 + soccomp_1 + female_1 + fam_econ_1",
       std, "A 卷：成癮 + 上網時數 + 上床時間 + FOMO + 社會比較")

corr_cols = {"addict_1": "網路成癮", "net_leisure_1": "休閒上網時數", "bedtime_1": "上床時間",
             "sleep_hrs_1": "睡眠時數", "fomo_1": "FOMO", "soccomp_1": "網路向上比較",
             "cyber_victim_1": "網路霸凌受害", "cesd_1": "國七憂鬱"}
cr = panel[list(corr_cols) + ["cesd_2", "selfharm_2", "grade_rank_2"]].corr().loc[list(corr_cols), ["cesd_2", "selfharm_2", "grade_rank_2"]]
cr.index = corr_cols.values(); cr.columns = ["國八憂鬱", "國八自傷意念", "國八成績排名(大=差)"]
print("\n== 國七變項與國八結果之相關 ==\n", cr.round(3).to_string())
cr.round(3).to_csv(TAB / "w1_w2_correlations.csv", encoding="utf-8-sig")
(ROOT / "data" / "processed").mkdir(exist_ok=True)
panel.to_pickle(ROOT / "data" / "processed" / "panel.pkl")
