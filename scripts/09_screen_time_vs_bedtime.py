"""螢幕時間 vs 上床時間：國七休閒上網時數與上床時間互相控制後，誰仍預測國八憂鬱？

延伸 01_eda.py 第 6 節的延宕迴歸。結果 = 國八 CESD（排除睡眠題，0–44），一律控制國七同量表、性別、家庭經濟；
WLS（HOUWGT）＋學校群集 SE。addict_1、net_leisure_1、bedtime_1、fomo_1、soccomp_1 以串接樣本（未加權）z 分數
標準化，係數 = 每 1 SD 的國八 CESD 分數。主樣本 = 國七有 A 卷者（含課後補填，與 01_eda 相同）。
  M1 全模型：國七憂鬱 + 成癮 + 休閒上網時數 + 上床時間 + FOMO + 網路向上比較 + 性別 + 家庭經濟；
     Wald 檢定 β上床 − β上網 = 0（兩係數各自是否顯著 ≠ 兩者有差異，比較要看差值）
  M2 只放休閒上網時數；M3 只放上床時間：各在自身完整樣本與 M1 樣本估計，分開「樣本不同」與「加入控制」
  逐步（M1 樣本）：M2 → +上床時間 → +成癮 → +FOMO、向上比較（= M1），看上網時數係數如何縮小
  穩健性：+數位發展分區、只用課堂 A 卷（03 主分析樣本，排除課後補填）、兩者皆是（= 03 慣例）、未加權
皆為關聯性推論；係數縮小與「經由晚睡／成癮」或「共同原因」皆相容，此設計無法區分。

執行：.venv/Scripts/python -I scripts/09_screen_time_vs_bedtime.py
輸出：outputs/screen_vs_bedtime/{tables,figures}/、outputs/screen_vs_bedtime/results.md
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tigps import data as D  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
OUT = ROOT / "outputs" / "screen_vs_bedtime"
FIG, TAB = OUT / "figures", OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

Y, Y1, COV = "cesd_nosleep_2", "cesd_nosleep_1", ["female_1", "fam_econ_1"]
Z = ["addict_1", "net_leisure_1", "bedtime_1", "fomo_1", "soccomp_1"]
LAB = {"net_leisure_1": "休閒上網時數", "bedtime_1": "上床時間", "addict_1": "網路成癮",
       "fomo_1": "FOMO", "soccomp_1": "網路向上比較"}
DIGI = ["d_pot", "d_start", "d_sprout"]   # 數位發展潛力／起步／萌動區（參照：成熟區）
BLUE, RED, INK, SEC, MUTED, GRID = "#2a78d6", "#d9432f", "#1f1f1f", "#52514e", "#898781", "#e6e4de"  # 同報告圖
MD: list[str] = []


def log(s=""):
    print(s)
    MD.append(s)


def fe(p):
    return "p<.001" if p < .001 else f"p={p:.3f}"


def cc(d, xs, cov=COV):
    return d.dropna(subset=[Y, Y1, *xs, *cov, "HOUWGT", "nschool_id"])


def fit(xs, d, cov=COV, weighted=True):
    """WLS（或未加權）＋學校群集 SE；d 須為 cc() 產生的完整資料樣本。"""
    w = d["HOUWGT"] if weighted else np.ones(len(d))
    f = smf.wls(f"{Y} ~ {' + '.join([Y1, *xs, *cov])}", d, weights=w)
    return f.fit(cov_type="cluster", cov_kwds={"groups": d["nschool_id"]})


def rows(res, d, model, sample, terms):
    ci = res.conf_int()
    return [dict(設定=sample, 模型=model, n=int(res.nobs), 學校數=d["nschool_id"].nunique(), 變項=LAB[t], term=t,
                 係數=res.params[t], CI下=ci.loc[t, 0], CI上=ci.loc[t, 1], p=res.pvalues[t]) for t in terms]


def diff(res, d, sample):
    t = res.t_test("bedtime_1 - net_leisure_1 = 0")
    lo, hi = np.asarray(t.conf_int()).ravel()
    return dict(設定=sample, n=int(res.nobs), 學校數=d["nschool_id"].nunique(), 對比="上床時間 − 休閒上網時數",
                差值=float(np.ravel(t.effect)[0]), CI下=lo, CI上=hi, p=float(np.ravel(t.pvalue)[0]))


def txt(r, k=3):
    return f"{r['係數']:+.{k}f} [{r['CI下']:.{k}f}, {r['CI上']:.{k}f}]，{fe(r['p'])}"


def cell(g, t):
    return txt(g[g.term == t].iloc[0]) if (g.term == t).any() else "—"


# ---------- 0. 資料 ----------
panel = D.load_panel()
p = panel.copy()
raw_sd = {c: p[c].std() for c in Z}
for c in Z:                                   # 與 01_eda 相同：串接樣本、未加權 z 分數
    p[c] = (p[c] - p[c].mean()) / p[c].std()
resp = pyreadstat.read_sav(str(D.W1_SAV), usecols=["nstudent_id", "response"])[0].set_index("nstudent_id")["response"]
p["ca_1"] = p["nstudent_id"].map(resp).isin([1, 3])          # 課堂作答 A 卷（1 共A附B、3 共A）；2 = 課後補填
for lvl, nm in zip(["數位發展潛力區", "數位發展起步區", "數位發展萌動區"], DIGI):
    p[nm] = (p["digi_level"] == lvl).astype(float).where(p["digi_level"].notna())
A = p[p["has_A_1"].astype(bool)]
CA = p[p["ca_1"]]

S1 = cc(A, Z)
S2, S3 = cc(A, ["net_leisure_1"]), cc(A, ["bedtime_1"])
log("# 螢幕時間 vs 上床時間（scripts/09_screen_time_vs_bedtime.py）\n")
log("結果 = 國八憂鬱 CESD（排除睡眠題，0–44）；皆控制國七同量表、性別、家庭經濟；WLS（HOUWGT）＋學校群集 SE（z 檢定）。"
    "預測變項為串接樣本 z 分數，係數 = 國七每多 1 SD，國八憂鬱高幾分。主樣本 = 國七有 A 卷者（含課後補填 A 卷，"
    "與 01_eda 相同；A/B 卷以學校為單位分派，非隨機）。\n")
log("## 0. 樣本與量尺\n")
log(f"- 串接 n={len(panel):,}；國七有 A 卷 n={len(A):,}（課堂 A 卷 {int(A['ca_1'].sum()):,}＋B 卷學校課後自願補填 "
    f"{int((~A['ca_1']).sum()):,}）。")
log(f"- 完整資料：M1（全模型）n={len(S1):,}（{S1['nschool_id'].nunique()} 校，其中補填者 {int((~S1['ca_1']).sum())} 人）；"
    f"M2（只放上網時數）自身樣本 n={len(S2):,}；M3（只放上床時間）自身樣本 n={len(S3):,}。")
log(f"- 1 SD 的原始單位：休閒上網時數 {raw_sd['net_leisure_1']:.2f} 小時（平日遊戲＋影音＋聊天＋瀏覽四項組中點加總，"
    f"各項可重疊，宜視為指數）；上床時間 {raw_sd['bedtime_1'] * 60:.0f} 分鐘；網路成癮 {raw_sd['addict_1']:.2f}（1–4）；"
    f"FOMO {raw_sd['fomo_1']:.2f}；網路向上比較 {raw_sd['soccomp_1']:.2f}。")
wc = lambda a, b: np.cov(S1[a], S1[b], aweights=S1["HOUWGT"])  # noqa: E731
r_lb, r_la = [(m := wc("net_leisure_1", x))[0, 1] / np.sqrt(m[0, 0] * m[1, 1]) for x in ("bedtime_1", "addict_1")]
log(f"- M1 樣本內加權相關：休閒上網時數 × 上床時間 r={r_lb:.2f}；休閒上網時數 × 網路成癮 r={r_la:.2f}"
    "（共線性使互相控制後的 CI 變寬）。\n")

# ---------- 1. 主模型 M1／M2／M3 ----------
coef, dif = [], []
r1 = fit(Z, S1)
coef += rows(r1, S1, "M1 全模型", "M1 樣本", Z)
dif.append(diff(r1, S1, "主樣本（全部 A 卷）"))
for nm, x, own in [("M2 只放上網時數", "net_leisure_1", S2), ("M3 只放上床時間", "bedtime_1", S3)]:
    coef += rows(fit([x], S1), S1, nm, "M1 樣本", [x])
    coef += rows(fit([x], own), own, nm, "自身完整樣本", [x])
coef = pd.DataFrame(coef)
d0 = dif[0]

log("## 1. 主模型\n")
log("| 模型 | 樣本 | n（校） | 休閒上網時數 | 上床時間 | 網路成癮 |")
log("|---|---|---|---|---|---|")
for (s, m), g in coef.groupby(["設定", "模型"], sort=False):
    log(f"| {m} | {s} | {g.n.iloc[0]:,}（{g.學校數.iloc[0]}） | {cell(g, 'net_leisure_1')} | {cell(g, 'bedtime_1')} | "
        f"{cell(g, 'addict_1')} |")
m1 = coef[coef.模型 == "M1 全模型"].set_index("term")
log(f"\nM1 其他係數：FOMO {txt(m1.loc['fomo_1'])}；網路向上比較 {txt(m1.loc['soccomp_1'])}。"
    f"上床時間的精確 p={m1.loc['bedtime_1', 'p']:.4f}（01_eda 輸出四捨五入至三位，顯示為 0.001）。")
log(f"\n**差值檢定（M1，Wald）：β上床 − β上網 = {d0['差值']:+.3f} [{d0['CI下']:.3f}, {d0['CI上']:.3f}]，{fe(d0['p'])}**"
    "——兩係數的差異未達顯著，不能據此說「上床時間比上網時數重要」。\n")

# ---------- 2. 逐步加入（M1 樣本） ----------
steps = [("① 只放上網時數（= M2）", ["net_leisure_1"]),
         ("② + 上床時間", ["net_leisure_1", "bedtime_1"]),
         ("③ + 網路成癮", ["net_leisure_1", "bedtime_1", "addict_1"]),
         ("④ + FOMO、向上比較（= M1）", Z)]
st = []
for nm, xs in steps:
    r = fit(xs, S1)
    st += [dict(步驟=nm, **rows(r, S1, nm, "M1 樣本", [t])[0], R2=r.rsquared)
           for t in ("net_leisure_1", "bedtime_1") if t in xs]
st = pd.DataFrame(st).drop(columns=["設定", "模型"])
log(f"## 2. 逐步加入控制（同一 M1 樣本 n={len(S1):,}）\n")
log("| 步驟 | 休閒上網時數 | 上床時間 | R² |")
log("|---|---|---|---|")
for nm, g in st.groupby("步驟", sort=False):
    log(f"| {nm} | {cell(g, 'net_leisure_1')} | {cell(g, 'bedtime_1')} | {g.R2.iloc[0]:.4f} |")
nl = st[st.term == "net_leisure_1"].set_index("步驟")["係數"]
s2 = st[(st.term == "net_leisure_1")].iloc[1]
log(f"\n上網時數係數由 {nl.iloc[0]:.3f} → 加入上床時間 {nl.iloc[1]:.3f}（−{1 - nl.iloc[1] / nl.iloc[0]:.0%}，仍 {fe(s2['p'])}）"
    f"→ 加入成癮 {nl.iloc[2]:.3f} → 全模型 {nl.iloc[3]:.3f}（累計 −{1 - nl.iloc[3] / nl.iloc[0]:.0%}）。"
    "只控制上床時間時，休閒上網時數仍有額外關聯；變得不顯著是在再加入網路成癮之後。\n")

# ---------- 3. 穩健性 ----------
VARIANTS = [("主樣本（全部 A 卷）", A, COV, True), ("+ 數位發展分區", A, COV + DIGI, True),
            ("只用課堂 A 卷", CA, COV, True), ("課堂 A 卷 + 數位分區（= 03 慣例）", CA, COV + DIGI, True),
            ("未加權（全部 A 卷）", A, COV, False)]
rob = []
for nm, data, cov, wt in VARIANTS:
    d = cc(data, Z, cov)
    r = fit(Z, d, cov, wt)
    rob += rows(r, d, "M1 全模型", nm, ["net_leisure_1", "bedtime_1", "addict_1"])
    rob += rows(fit(["net_leisure_1"], d, cov, wt), d, "M2 只放上網時數", nm, ["net_leisure_1"])
    rob += rows(fit(["bedtime_1"], d, cov, wt), d, "M3 只放上床時間", nm, ["bedtime_1"])
    if nm != VARIANTS[0][0]:
        dif.append(diff(r, d, nm))
rob, dif = pd.DataFrame(rob), pd.DataFrame(dif)
log("## 3. 穩健性（各設定皆在該設定的 M1 完整樣本上估計 M1、M2、M3）\n")
log("| 設定 | n（校） | M2 上網時數（單獨） | M1 上網時數 | M1 上床時間 | M1 網路成癮 | 差值 上床−上網 |")
log("|---|---|---|---|---|---|---|")
for nm, g in rob.groupby("設定", sort=False):
    m1g, m2g = g[g.模型 == "M1 全模型"], g[g.模型 == "M2 只放上網時數"]
    dd = dif[dif.設定 == nm].iloc[0]
    log(f"| {nm} | {g.n.iloc[0]:,}（{g.學校數.iloc[0]}） | {cell(m2g, 'net_leisure_1')} | {cell(m1g, 'net_leisure_1')} | "
        f"{cell(m1g, 'bedtime_1')} | {cell(m1g, 'addict_1')} | {dd['差值']:+.3f} [{dd['CI下']:.3f}, {dd['CI上']:.3f}]，"
        f"{fe(dd['p'])} |")
sup = cc(A[~A["ca_1"]], Z)
rs = fit(Z, sup)
alt = rob.query("模型 == 'M1 全模型' and 設定 != @VARIANTS[0][0]")
pl, pb = alt[alt.term == "net_leisure_1"]["p"], rob.query("模型 == 'M1 全模型' and term == 'bedtime_1'")["p"]
log(f"\n主樣本 M1 的上網時數 p=.088 對設定敏感：加入數位分區、排除課後補填者或不加權，p 落在 {pl.min():.3f}–{pl.max():.3f}。"
    f"原因之一是 {len(sup)} 名課後補填者（{sup['nschool_id'].nunique()} 校；B 卷學校自願補填，"
    f"「過去一個月」作息可能落在暑假）自成一組時上網時數斜率為 {rs.params['net_leisure_1']:+.2f} "
    f"[{rs.conf_int().loc['net_leisure_1', 0]:.2f}, {rs.conf_int().loc['net_leisure_1', 1]:.2f}]，把合併估計往下拉。"
    f"上床時間（各設定 p≤{pb.max():.3f}）與「兩係數差異不顯著」（p={dif['p'].min():.3f}–{dif['p'].max():.3f}）"
    "則在所有設定下都成立。\n")

coef = pd.concat([coef, rob], ignore_index=True)
coef.round(4).to_csv(TAB / "coefficients.csv", index=False, encoding="utf-8-sig")
st.round(4).to_csv(TAB / "stepwise_leisure.csv", index=False, encoding="utf-8-sig")
dif.round(4).to_csv(TAB / "difference_test.csv", index=False, encoding="utf-8-sig")

# ---------- 4. 解讀 ----------
def g(m, t, s="M1 樣本"):
    return coef.query("模型 == @m and term == @t and 設定 == @s").iloc[0]


m2, m2own = g("M2 只放上網時數", "net_leisure_1"), g("M2 只放上網時數", "net_leisure_1", "自身完整樣本")
m3 = g("M3 只放上床時間", "bedtime_1")
a, b, c_ = g("M1 全模型", "net_leisure_1"), g("M1 全模型", "bedtime_1"), g("M1 全模型", "addict_1")
VS = "課堂 A 卷 + 數位分區（= 03 慣例）"
a2, d2 = g("M1 全模型", "net_leisure_1", VS), dif[dif.設定 == VS].iloc[0]
log("## 4. 解讀（皆為關聯性推論）\n")
log(f"1. **單獨看螢幕時間，它與一年後的憂鬱相關。** 控制國七憂鬱、性別、家庭經濟後，國七平日休閒上網時數每多 1 SD"
    f"（約 {raw_sd['net_leisure_1']:.1f} 小時），國八憂鬱高 {m2['係數']:.2f} 分 [{m2['CI下']:.2f}, {m2['CI上']:.2f}]"
    f"（{fe(m2['p'])}，n={m2['n']:,}；自身完整樣本 n={m2own['n']:,} 為 {m2own['係數']:.2f} [{m2own['CI下']:.2f}, "
    f"{m2own['CI上']:.2f}]）。上床時間單獨放入時為每 1 SD（約 {raw_sd['bedtime_1'] * 60:.0f} 分鐘）"
    f"{m3['係數']:.2f} [{m3['CI下']:.2f}, {m3['CI上']:.2f}]。")
log(f"2. **只控制上床時間時，休閒上網時數仍有額外關聯**（{s2['係數']:.2f} [{s2['CI下']:.2f}, {s2['CI上']:.2f}]，"
    f"{fe(s2['p'])}，縮小約兩成）。「控制上床時間後偵測不到螢幕的額外關聯」只適用於深夜螢幕（晚間日誌 23:00 後、"
    "晚間類型；見 02、03），不適用於平日休閒上網總時數。")
log(f"3. **全模型中，上床時間的關聯仍清楚，上網時數則不確定。** 同時放入成癮、FOMO、向上比較後：上床時間 "
    f"{b['係數']:.3f} [{b['CI下']:.3f}, {b['CI上']:.3f}]（{fe(b['p'])}）；休閒上網時數 {a['係數']:.3f} "
    f"[{a['CI下']:.3f}, {a['CI上']:.3f}]（{fe(a['p'])}）；網路成癮 {c_['係數']:.3f} [{c_['CI下']:.3f}, "
    f"{c_['CI上']:.3f}]（{fe(c_['p'])}）。上網時數係數累計縮小約一半，與「螢幕時間部分經由晚睡、失控使用與一年後"
    "憂鬱相連」相容，也與「有共同原因」相容，兩波資料無法區分。而且這個「不顯著」是邊緣的：在 03 的主分析慣例下"
    f"（課堂 A 卷＋數位分區）上網時數為 {a2['係數']:.3f} [{a2['CI下']:.3f}, {a2['CI上']:.3f}]（{fe(a2['p'])}）。")
log(f"4. **這不是「上床時間比螢幕時間更重要」的檢定結果。** 兩係數之差 {d0['差值']:+.3f} [{d0['CI下']:.3f}, "
    f"{d0['CI上']:.3f}]，{fe(d0['p'])}（03 慣例下 {d2['差值']:+.3f}，{fe(d2['p'])}），未達顯著；一個顯著、一個不顯著，"
    f"不等於兩者顯著不同。上網時數的 CI 上界達 {a['CI上']:.2f} 分，也不能解讀為螢幕時間無害。")
log("5. **把「就寢時間」列為優先著力點，依據是其他分析的一致方向，而非本表的大小比較：** 深夜螢幕與上床時間的巢套檢定"
    "不對稱（同時放入：上床時間每晚 1 小時 +0.50 分 p=.003，深夜螢幕 +0.60 分 p=.127；加入上床時間後晚間類型不再"
    "偵測到額外關聯）、兩波 CLPM 的方向（上床時間 → 一年後憂鬱 β=0.065，反向 β=0.003），以及各分析間的一致性。"
    "螢幕是熬夜的常見原因之一，但不是唯一（寫作業、補習也會晚睡）；因此教育上的建議是把「用多久」延伸到「幾點收」，"
    "而不是只管其中一項。")
log("6. 效果皆小（每 1 SD 約 0.3–0.7 分，0–44 分量尺）；A 卷學校非隨機分派，未調整的共同原因（如家庭作息、課業壓力）"
    "仍可能解釋部分關聯。")
(OUT / "results.md").write_text("\n".join(MD) + "\n", encoding="utf-8")

# ---------- 5. 係數圖：主樣本（實心）vs 03 慣例（空心） ----------
lab = ["只放上網時數（M2）", "只放上床時間（M3）", "同時放入（M1）：上網時數", "同時放入（M1）：上床時間",
       "差值：上床 − 上網（M1）"]
spec = [("M2 只放上網時數", "net_leisure_1", RED, "o"), ("M3 只放上床時間", "bedtime_1", BLUE, "s"),
        ("M1 全模型", "net_leisure_1", RED, "o"), ("M1 全模型", "bedtime_1", BLUE, "s")]
ys = np.array([4.4, 3.4, 1.9, 0.9, -0.6])
fig, ax = plt.subplots(figsize=(7.2, 3.4))
for k, (setting, off, fill) in enumerate([("主樣本（全部 A 卷）", .15, True), (VS, -.15, False)]):
    est = [coef.query("設定 == @setting and 模型 == @m and term == @t").iloc[0] for m, t, _, _ in spec]
    dd = dif[dif.設定 == setting].iloc[0]
    est.append(dict(係數=dd["差值"], CI下=dd["CI下"], CI上=dd["CI上"], p=dd["p"]))
    for r, y, (col, mk) in zip(est, ys + off, [(c, m) for _, _, c, m in spec] + [(SEC, "D")]):
        ax.plot([r["CI下"], r["CI上"]], [y, y], color=col, lw=2 if fill else 1.2, solid_capstyle="round")
        ax.plot(r["係數"], y, mk, ms=7, mew=1.5, color=col, mfc=col if fill else "white", mec="white" if fill else col)
        ax.text(r["CI上"] + .03, y, f"{r['係數']:.2f}（{fe(r['p'])}）", va="center", fontsize=7.5, color=SEC)
ax.axvline(0, color=MUTED, lw=.8)
for yy in (2.65, 0.15):
    ax.axhline(yy, color=GRID, lw=.8)
ax.set_yticks(ys, lab, fontsize=8.5)
ax.set_xlim(-.35, 1.45)
ax.set_ylim(-1.1, 4.9)
ax.set_xlabel("國七每多 1 SD → 國八憂鬱分數（0–44），95% CI", color=SEC, fontsize=8.5)
ax.set_title("休閒上網時數與上床時間：單獨 vs 互相控制（控制國七憂鬱、性別、家庭經濟）", fontsize=9.5, color=INK, loc="left")
n_vs = rob[rob.設定 == VS].n.iloc[0]
ax.plot([], [], "o", color=SEC, ms=6, label=f"實心：全部 A 卷 n={len(S1):,}")
ax.plot([], [], "o", color=SEC, mfc="white", ms=6, label=f"空心：課堂 A 卷＋數位分區 n={n_vs:,}")
ax.legend(loc="upper center", bbox_to_anchor=(.4, -.17), ncol=2, fontsize=7.5, frameon=False)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", colors=SEC, labelsize=8)
fig.tight_layout()
fig.savefig(FIG / "screen_vs_bedtime_coef.png", dpi=200, bbox_inches="tight")
print(f"\n輸出：{OUT}")
