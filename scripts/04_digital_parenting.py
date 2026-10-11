"""數位教養：哪一種「管網路」的方式，和較早睡、較少成癮、較少憂鬱有關？

主分析：國八（W2）同期、全樣本；穩健性：控制國七同一結果、國七（W1）B 卷數位教養 → 國八結果。
1. 測量：W2 父/母 12 題 → 平行分析 + ML 因素分析（promax 斜交）；CFA（semopy；含 SRMR、殘差相關、
   半樣本驗證、W1 同義題獨立驗證）；W1 同義題 EFA；Cronbach α
2. 主效果：加權（HOUWGT）WLS、學校叢集標準誤、標準化 β、BH-FDR。
   主模型＝「一次放一種數位教養 + 一般親子（溫暖/敵意/凝聚）+ 共變項」，不互相控制其他數位教養：
   責罵/擔憂等題目可能是孩子問題上網的「結果」（碰撞點）或規則的中介，控制它們會扭曲規則的關聯。
   互相控制版本只當敏感度並標註「可能有碰撞偏誤」。「鼓勵上網」雖與陪伴題同一因素，但與成癮的關聯方向相反，
   迴歸中分開放入（六構面＝data.py PAR_W2）；鼓勵併入陪伴的五構面版本作敏感度。
3. 調節：成癮 / 上床時間 × 教養 → 憂鬱（同期、半前瞻、W1 前瞻）；簡單斜率、Johnson–Neyman
4. 反向因果：W1→W2 交叉延宕（親→子 vs 子→親；堆疊方程式 + 叢集穩健共變異檢定兩路徑是否相等）

執行：.venv/Scripts/python -I scripts/04_digital_parenting.py
輸出：outputs/parenting/figures/*.png、outputs/parenting/tables/*.csv、outputs/parenting/results.md
"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import semopy
import statsmodels.api as smapi
import statsmodels.formula.api as smf
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats
from sklearn.decomposition import FactorAnalysis
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tigps import data as D  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
OUT = ROOT / "outputs" / "parenting"
FIG, TAB = OUT / "figures", OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
SEED = 2026
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#2b2b2a", "#52514e"
LOG: list[str] = []


def log(*s):
    txt = " ".join(str(x) for x in s)
    print(txt)
    LOG.append(txt)


def save(df, name, **kw):
    df.to_csv(TAB / name, encoding="utf-8-sig", **kw)


def cronbach(df):
    df = df.dropna()
    k = df.shape[1]
    return k / (k - 1) * (1 - df.var().sum() / df.sum(axis=1).var()) if k > 1 else np.nan


def wms(s, w):
    m = np.average(s, weights=w)
    return m, np.sqrt(np.average((s - m) ** 2, weights=w))


def wz(s, w):
    """加權 z 分數（在分析樣本內）。"""
    m, sd = wms(s, w)
    return (s - m) / sd


def wcorr(a, b, w):
    ma, mb = np.average(a, weights=w), np.average(b, weights=w)
    return np.average((a - ma) * (b - mb), weights=w) / np.sqrt(
        np.average((a - ma) ** 2, weights=w) * np.average((b - mb) ** 2, weights=w))


def pfmt(p):
    return "p<.001" if p < .001 else f"p={p:.3f}"


def qfmt(q):
    return "q<.001" if q < .001 else f"q={q:.3f}"


# ---------- 0. 資料與構面定義 ----------
raw1, _ = D.read_wave(D.W1_SAV)
raw2, _ = D.read_wave(D.W2_SAV)
panel = D.load_panel()
L2 = "abcdefghijkl"
ITEM2 = {"a": "規定何時可上網", "b": "限制上網時間長短", "c": "聊網路趣聞", "d": "陪查資料/寫作業",
         "e": "鼓勵上網", "f": "討論使用心得", "g": "擔心健康", "h": "擔心課業",
         "i": "問在網上做什麼", "j": "查看瀏覽紀錄", "k": "因上網責罵", "l": "因網購起衝突"}
ITEM1X = {"b": "禁止某些網站", "e": "一起玩遊戲/追劇", "i": "認為擅長網路對未來有幫助", "k": "擔心網路風險",
          "n": "知道網友是誰", "p": "上網時碎碎念"}       # W1 獨有題
W1SYN = dict(zip("acdfghjlmoqr", L2))                          # W1 題 → 同義 W2 題
ITEM1 = {c: ITEM2[W1SYN[c]] if c in W1SYN else ITEM1X[c] for c in "abcdefghijklmnopqr"}
DP2 = {"restrict": "ab", "active": "cdf", "encourage": "e", "worry": "gh", "monitor": "ij", "conflict": "kl"}  # = data.py PAR_W2
DP1 = {"restrict": "ac", "active": "dfh", "encourage": "g", "worry": "jl", "monitor": "mo", "conflict": "qr"}  # W1 取與 W2 同義題
MRG2, MRG1 = {"active5": "cdef"}, {"active5": "dfgh"}          # EFA 版：鼓勵併入陪伴（敏感度）
DPK = list(DP2)
DPLAB = {"restrict": "規範限制", "active": "陪伴引導", "encourage": "鼓勵上網", "worry": "擔憂叮嚀", "monitor": "監看",
         "conflict": "衝突責罵", "active5": "陪伴引導(含鼓勵)", "warmth": "溫暖支持", "hostile": "敵意/冷漠",
         "cohesion": "家庭凝聚"}


def items(df, father, mother, letters):
    """逐題取父、母可用值平均（與 data.py 構面同規則；5/負值=不適用/遺漏）。"""
    f = df[[father + c for c in letters]].where(lambda x: x <= 4).set_axis(list(letters), axis=1)
    m = df[[mother + c for c in letters]].where(lambda x: x <= 4).set_axis(list(letters), axis=1)
    return pd.concat([f, m]).groupby(level=0).mean(), f, m


isB = raw1["response"].isin([1, 2, 4])
it2, fa2, mo2 = items(raw2, "bs20a", "bs20b", L2)
it1, _, _ = items(raw1[isB], "as28", "as29", "abcdefghijklmnopqr")
dp2 = D.parenting(raw2, "bs20a", "bs20b", {**DP2, **MRG2}).rename(columns=lambda c: c.replace("par_", "dp_") + "_2")
dp1 = D.parenting(raw1, "as28", "as29", {**DP1, **MRG1}).rename(columns=lambda c: c.replace("par_", "dp_") + "_1")
dp1.loc[~isB] = np.nan                                         # 僅 B 卷
dp2["nstudent_id"], dp1["nstudent_id"] = raw2["nstudent_id"], raw1["nstudent_id"]
panel = panel.merge(dp2, on="nstudent_id", how="left").merge(dp1, on="nstudent_id", how="left")
log("# 數位教養：哪一種管法有幫助？（04_digital_parenting.py）\n")
log("## 0. 本版修訂（回應審查）\n\n"
    "- 主模型改為「一次一種數位教養 + 一般親子 + 共變項」；原「全部互相控制」模型降為敏感度並標註可能有碰撞偏誤（2g、2i、圖 S2、S3）。\n"
    "- 迴歸中「鼓勵上網」與陪伴引導分開（六構面＝data.py PAR_W2）；EFA 版「鼓勵併入陪伴」改為敏感度。\n"
    "- 國七 B 卷上床/睡眠模型改名「W1教養_未控W1作息」；新增 A+B 卷同一樣本比較（bootstrap 差值，2e）與時序負對照（2f）。\n"
    "- 交叉延宕加入鼓勵上網，新增兩方向差異的堆疊方程式 Wald 檢定，並註明單一構面、兩波的限制。\n"
    "- CFA 加 SRMR、殘差相關、半樣本驗證與國七獨立驗證，配適改述為「不佳」；新增國七 EFA 輸出。\n"
    "- 四格圖標籤改為「有責罵／幾乎不責罵」，標題以責罵為主軸並加碰撞點警語；調節分析改為與主模型一致的單一教養設定。")
log(f"串接樣本 n={len(panel)}；W2 數位教養有效 n={panel['dp_restrict_2'].notna().sum()}；"
    f"W1 B 卷數位教養有效 n={panel['dp_restrict_1'].notna().sum()}")
log("註：codebook 中「不適用」實際編碼為 -4（已由 read_wave 設為遺漏），值 5 未出現；兩者皆排除。")
dif = max((panel[f"dp_{k}_2"] - panel[f"par_{k}_2"]).abs().max() for k in DPK)
log(f"迴歸用六構面：W2 與 data.py PAR_W2 完全相同（最大差 {dif:.1e}）；W1 改用與 W2 同義的題目，與 data.py PAR_W1 的差異："
    "規範限制 ac（data.py abc，去掉 b 禁止某些網站）、擔憂叮嚀 jl（jkl，去掉 k 擔心網路風險）、衝突責罵 qr（pqr，去掉 p 碎碎念）；"
    "陪伴引導 dfh、鼓勵 g、監看 mo 相同。")

# ---------- 1. 測量：EFA / CFA / α ----------
log("\n## 1. 數位教養構面驗證（W2 12 題，父母逐題平均）\n")


def promax(A, k=4):
    """varimax → promax 斜交轉軸；回傳 pattern loadings 與因素相關。"""
    p, nf = A.shape
    R, d = np.eye(nf), 0
    for _ in range(1000):
        Lr = A @ R
        u, s, vt = np.linalg.svd(A.T @ (Lr ** 3 - Lr @ np.diag((Lr ** 2).sum(0)) / p))
        R = u @ vt
        if s.sum() < d * (1 + 1e-10):
            break
        d = s.sum()
    V = A @ R
    Vn = V / np.sqrt((V ** 2).sum(1, keepdims=True))
    U = np.linalg.lstsq(Vn, np.abs(Vn) ** k * np.sign(Vn), rcond=None)[0]
    U = U @ np.diag(np.sqrt(np.diag(np.linalg.inv(U.T @ U))))
    P, phi = V @ U, np.linalg.inv(U.T @ U)
    s = np.sign(P.sum(0))
    return P * s, phi * np.outer(s, s)


def efa(X, nf):
    Z = (X - X.mean()) / X.std()
    return promax(FactorAnalysis(n_components=nf, svd_method="lapack").fit(Z).components_.T)


def parallel(X, rng, reps=200):
    Z = (X - X.mean()) / X.std()
    eig = np.linalg.eigvalsh(Z.corr())[::-1]
    sim = np.array([np.linalg.eigvalsh(np.corrcoef(rng.standard_normal(Z.shape), rowvar=False))[::-1]
                    for _ in range(reps)])
    pa95 = np.percentile(sim, 95, axis=0)
    return eig, pa95, int(np.argmax(eig < pa95))


def where_e(P, lab_e, lab_act):
    """回傳鼓勵題的最大負荷，以及該因素上陪伴題的負荷（確認是否同一因素）。"""
    j = int(np.argmax(P.loc[lab_e].values))
    return P.loc[lab_e].iloc[j], P.iloc[:, j].loc[lab_act].round(2).tolist()


rng = np.random.default_rng(SEED)
X2 = it2.dropna()
eig, pa95, n_pa = parallel(X2, rng)
log(f"EFA 樣本 n={len(X2)}（兩位家長皆不適用/遺漏者排除）。特徵值前 5：{np.round(eig[:5], 2).tolist()}；"
    f"平行分析 95% 門檻：{np.round(pa95[:5], 2).tolist()} → 建議保留 {n_pa} 個因素")
EFA = {}
for nf in (3, 4, 5):
    P, phi = efa(X2, nf)
    EFA[nf] = pd.DataFrame(P, index=[f"{c} {ITEM2[c]}" for c in L2], columns=[f"F{j+1}" for j in range(nf)])
    save(EFA[nf].round(3), f"efa_loadings_{nf}f.csv")
    save(pd.DataFrame(phi, index=EFA[nf].columns, columns=EFA[nf].columns).round(3), f"efa_phi_{nf}f.csv")
log("4 因素 promax pattern loadings：\n" + EFA[4].round(2).to_string())
ACT2 = [f"{c} {ITEM2[c]}" for c in "cdf"]
e2_load, _ = where_e(EFA[4], "e 鼓勵上網", ACT2)

# 半樣本：A 半做 EFA、B 半做 CFA（避免同一份資料探索又驗證）
perm = rng.permutation(len(X2))
X2a, X2b = X2.iloc[perm[: len(X2) // 2]], X2.iloc[perm[len(X2) // 2:]]
Pa = pd.DataFrame(efa(X2a, 4)[0], index=EFA[4].index)
ea_load, ea_act = where_e(Pa, "e 鼓勵上網", ACT2)
log(f"半樣本 A（n={len(X2a)}）重做 4 因素 EFA：鼓勵上網最大負荷 {ea_load:.2f}，同因素上陪伴題 c/d/f 負荷 {ea_act}")

# W1 B 卷：12 題同義題 + 全部 18 題 EFA（獨立施測時點）
X1 = it1[list(W1SYN)].dropna().rename(columns=W1SYN)
P1, _ = efa(X1, 4)
efa_w1 = pd.DataFrame(P1, index=[f"{w1}→{w2} {ITEM2[w2]}" for w1, w2 in W1SYN.items()],
                      columns=[f"F{j+1}" for j in range(4)])
save(efa_w1.round(3), "efa_loadings_w1_12items_4f.csv")
X1all = it1.dropna()
_, _, n_pa1 = parallel(X1all, rng)
P1a, _ = efa(X1all, max(n_pa1, 4))
efa_w1a = pd.DataFrame(P1a, index=[f"{c} {ITEM1[c]}" for c in X1all.columns],
                       columns=[f"F{j+1}" for j in range(P1a.shape[1])])
save(efa_w1a.round(3), "efa_loadings_w1_18items.csv")
e1_load, e1_act = where_e(efa_w1, f"g→e {ITEM2['e']}", [f"{c}→{W1SYN[c]} {ITEM2[W1SYN[c]]}" for c in "dfh"])
e1a_load, e1a_act = where_e(efa_w1a, f"g {ITEM1['g']}", [f"{c} {ITEM1[c]}" for c in "dfh"])
log(f"W1 B 卷同義 12 題 4 因素 EFA（n={len(X1)}）：鼓勵上網(g) 最大負荷 {e1_load:.2f}，同因素陪伴題 d/f/h {e1_act}；"
    f"全部 18 題（n={len(X1all)}，平行分析建議 {n_pa1} 因素，取 {P1a.shape[1]}）：g 負荷 {e1a_load:.2f}，同因素 d/f/h {e1a_act}"
    "（tables/efa_loadings_w1_*.csv）")

ln = {k: f"p{k}" for k in L2}
CFA = {
    "M1 單一因素": {"G": L2},
    "M3 EFA三因素(控制/陪伴/侵入)": {"CTL": "abgh", "ENG": "cdef", "INT": "ijkl"},
    "M4 EFA四因素(i 歸擔憂、j 歸衝突)": {"RES": "ab", "ENG": "cdef", "WOR": "ghi", "INT": "jkl"},
    "M5 理論五因素(鼓勵併入陪伴)": {"RES": "ab", "ACT": "cdef", "WOR": "gh", "MON": "ij", "CON": "kl"},
}


def cfa(spec, data):
    desc = "\n".join(f"{f} =~ " + " + ".join(ln[c] for c in its) for f, its in spec.items())
    mdl = semopy.Model(desc)
    mdl.fit(data.set_axis([ln[c] for c in data.columns], axis=1))
    st = semopy.calc_stats(mdl).T["Value"]
    S, Sig = mdl.mx_cov, mdl.calc_sigma()[0]
    sd = np.sqrt(np.diag(S))
    Rres = (S - Sig) / np.outer(sd, sd)                       # 標準化殘差（相關尺度）
    srmr = np.sqrt(np.mean(Rres[np.tril_indices_from(Rres)] ** 2))
    n = len(data)
    return mdl, Rres, dict(n=n, df=int(st["DoF"]), chi2=st["chi2"], CFI=st["CFI"], TLI=st["TLI"], RMSEA=st["RMSEA"],
                           SRMR=srmr, BIC_rel=st["chi2"] - st["DoF"] * np.log(n))   # 相對 BIC（同資料比較用）


rows, m5, m5res = [], None, None
DATS = [("W2 父母平均", X2), ("W2 父親", fa2.dropna()), ("W2 母親", mo2.dropna()),
        ("W2 驗證半樣本B", X2b), ("W1 B卷同義題", X1)]
for name, spec in CFA.items():
    for who, dat in DATS:
        mdl, Rres, st = cfa(spec, dat)
        rows.append({"模型": name, "資料": who, **st})
        if name.startswith("M5") and who == "W2 父母平均":
            m5, m5res = mdl, Rres
cfa_tab = pd.DataFrame(rows).round(3)
save(cfa_tab, "cfa_fit.csv", index=False)
for who in ("W2 父母平均", "W2 驗證半樣本B", "W1 B卷同義題"):
    log(f"\nCFA 配適：{who}（ML；BIC_rel = χ² − df·ln N，越小越好）：\n"
        + cfa_tab[cfa_tab["資料"] == who].drop(columns="資料").to_string(index=False))
est = m5.inspect(std_est=True)
lat = est[(est.op == "~~") & (est.lval != est.rval) & est.lval.isin(["RES", "ACT", "WOR", "MON", "CON"])]
lat = lat[["lval", "rval", "Est. Std"]].rename(columns={"Est. Std": "r"})
lat[["lval", "rval"]] = np.sort(lat[["lval", "rval"]].values, axis=1)   # semopy 配對順序隨雜湊而變，固定排序
lat = lat.sort_values(["lval", "rval"]).reset_index(drop=True)
save(lat.round(3), "cfa_m5_latent_corr.csv", index=False)
load5 = est[est.op == "~"][["lval", "rval", "Est. Std"]]
obs = [v[1:] for v in m5.vars["observed"]]
ri, rj = np.tril_indices(len(obs), -1)
resid = pd.DataFrame({"題1": [f"{obs[i]} {ITEM2[obs[i]]}" for i in ri], "題2": [f"{obs[j]} {ITEM2[obs[j]]}" for j in rj],
                      "殘差相關": m5res[ri, rj]}).assign(a=lambda t: t["殘差相關"].abs()).sort_values("a", ascending=False)
resid = resid.drop(columns="a").reset_index(drop=True)
save(resid.round(3), "cfa_m5_residual_corr.csv", index=False)
log("\nM5 標準化負荷量範圍：" + f"{load5['Est. Std'].min():.2f}–{load5['Est. Std'].max():.2f}；潛在相關最高："
    + ", ".join(f"{a}–{b} {r:.2f}" for a, b, r in lat.sort_values("r", ascending=False).head(3).values))
log("M5 局部不適配（標準化殘差相關絕對值前 6）：\n" + resid.head(6).round(3).to_string(index=False))
m5r = cfa_tab[(cfa_tab["模型"].str.startswith("M5")) & (cfa_tab["資料"] == "W2 父母平均")].iloc[0]
m4r = cfa_tab[(cfa_tab["模型"].str.startswith("M4")) & (cfa_tab["資料"] == "W2 父母平均")].iloc[0]
m5b = cfa_tab[(cfa_tab["模型"].str.startswith("M5")) & (cfa_tab["資料"] == "W2 驗證半樣本B")].iloc[0]
m5w1 = cfa_tab[(cfa_tab["模型"].str.startswith("M5")) & (cfa_tab["資料"] == "W1 B卷同義題")].iloc[0]
log(f"判讀：M5 CFI={m5r.CFI:.3f}、TLI={m5r.TLI:.3f}、RMSEA={m5r.RMSEA:.3f}、SRMR={m5r.SRMR:.3f}——依慣例（RMSEA>.10、TLI<.90）屬配適不佳；"
    "RMSEA 偏高部分來自把 4 點類別題當連續變項、多數構面只有 2 題（df 小），以及 i/j 題跨構面的局部不適配。"
    "因此五/六構面視為「實用分組」，主要依據 EFA 型態與 α，而非 CFA 配適。")

# α：迴歸用六構面（W2、W1 同義題）＋ EFA 版陪伴(含鼓勵)
arow = []
for ver, mp, it, sfx in [("W2 六構面（迴歸用；=data.py）", {**DP2, **MRG2}, it2, "_2"),
                         ("W1 同義題六構面（B 卷）", {**DP1, **MRG1}, it1, "_1")]:
    for k, its in mp.items():
        v = panel[f"dp_{k}{sfx}"]
        arow.append({"版本": ver, "構面": k, "題目": its, "題數": len(its), "α": cronbach(it[list(its)]),
                     "加權平均": np.average(v.dropna(), weights=panel.loc[v.notna(), "HOUWGT"].fillna(1)), "SD": v.std()})
alpha = pd.DataFrame(arow).round(3)
save(alpha, "reliability_alpha.csv", index=False)
log("\nCronbach α（鼓勵為單題，無 α）：\n" + alpha.to_string(index=False))
stab = {k: panel[[f"dp_{k}_1", f"dp_{k}_2"]].corr().iat[0, 1] for k in DPK}
log("數位教養一年穩定性 r(W1,W2)：" + ", ".join(f"{DPLAB[k]} {v:.2f}" for k, v in stab.items()))
cc = panel[[f"dp_{k}_2" for k in DPK] + ["warmth_2", "hostile_2", "cohesion_2"]].corr()
cc.index = cc.columns = [DPLAB[k] for k in DPK + ["warmth", "hostile", "cohesion"]]
save(cc.round(3), "composite_corr_w2.csv")
log(f"r(陪伴 cdf, 鼓勵 e)={panel[['dp_active_2', 'dp_encourage_2']].corr().iat[0, 1]:.2f}；"
    f"r(規範限制, 衝突責罵)={panel[['dp_restrict_2', 'dp_conflict_2']].corr().iat[0, 1]:.2f}")

# EFA 負荷量熱圖（附錄用）
cmap = LinearSegmentedColormap.from_list("div", [BLUE, "#f0efec", "#e34948"])
fig, ax = plt.subplots(figsize=(5.6, 5.2))
GRP = {"a": "規範限制", "b": "規範限制", "c": "陪伴引導", "d": "陪伴引導", "e": "陪伴引導", "f": "陪伴引導",
       "g": "擔憂叮嚀", "h": "擔憂叮嚀", "i": "擔憂叮嚀", "j": "監看衝突", "k": "監看衝突", "l": "監看衝突"}
E = EFA[4].copy()                                              # 依負荷最高題目命名因素並排序
E.columns = [GRP[E.index[np.argmax(E[c].values)][0]] for c in E.columns]
E = E[[c for c in ["規範限制", "陪伴引導", "擔憂叮嚀", "監看衝突"] if c in E.columns]]
im = ax.imshow(E.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(4), E.columns)
ax.set_yticks(range(12), E.index)
for i in range(12):
    for j in range(4):
        v = E.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(v) > .6 else INK, fontweight="bold" if abs(v) >= .4 else "normal")
for yy in (1.5, 5.5, 7.5, 9.5):
    ax.axhline(yy, color="white", lw=2)
ax.set_title("國八數位教養 12 題探索性因素分析\n（ML + promax，4 因素 pattern loadings）", fontsize=11)
fig.colorbar(im, ax=ax, shrink=.7, label="負荷量")
fig.tight_layout()
fig.savefig(FIG / "fig_s1_efa_loadings.png", dpi=200)
plt.close(fig)


# ---------- 2. 主效果 ----------
def fit(df, y, xs, z, cat=("digi_level",), extra=""):
    """加權 WLS + 學校叢集 SE；z 中變項（含 y）在分析樣本內加權標準化。"""
    xs = list(dict.fromkeys(xs))
    cols = list(dict.fromkeys([y] + xs + list(cat) + ["HOUWGT", "nschool_id"]))
    d = df[cols].dropna().copy()
    for c in dict.fromkeys(z):
        d[c] = wz(d[c], d["HOUWGT"])
    rhs = " + ".join(xs + [f"C({c})" for c in cat]) + extra
    return smf.wls(f"{y} ~ {rhs}", d, weights=d["HOUWGT"]).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["nschool_id"])[0]}), d


def tidy(res, terms, **tags):
    ci = res.conf_int()
    return pd.DataFrame([{**tags, "term": t, "beta": res.params[t], "se": res.bse[t], "lo": ci.loc[t, 0],
                          "hi": ci.loc[t, 1], "p": res.pvalues[t], "n": int(res.nobs)} for t in terms])


def fdr(df, col="p"):
    df = df.copy()
    df["q"] = multipletests(df[col], method="fdr_bh")[1]
    return df


OUTC = {"bedtime_2": "上床時間", "sleep_hrs_2": "上課日睡眠時數", "addict_2": "網路成癮", "cesd_nosleep_2": "憂鬱（不含睡眠題）"}
LAG = {"bedtime_2": "bedtime_1", "sleep_hrs_2": "sleep_hrs_1", "addict_2": "addict_1", "cesd_nosleep_2": "cesd_nosleep_1"}
DPV2, DPV1 = [f"dp_{k}_2" for k in DPK], [f"dp_{k}_1" for k in DPK]
DPV2_5 = [f"dp_{k}_2" for k in ("restrict", "active5", "worry", "monitor", "conflict")]
GEN2, GEN1 = ["warmth_2", "hostile_2", "cohesion_2"], ["warmth_1", "hostile_1", "cohesion_1"]
COV2 = ["female_2", "subj_ses_2", "parents_married_cohab_2"]
COV1 = ["female_1", "fam_econ_1", "parents_married_cohab_1"]
Z2, Z1 = ["subj_ses_2"], ["fam_econ_1"]
SLEEPY = ("bedtime_2", "sleep_hrs_2")

log("\n## 2. 主效果：數位教養 → 國八作息/成癮/憂鬱（標準化 β）\n")
log("主模型＝一次只放一種數位教養，並控制一般親子（溫暖/敵意/凝聚）與共變項（性別、主觀社經、父母婚姻、數位發展區）；"
    "一般親子三列取自「一般親子三項 + 共變項」模型。不互相控制其他數位教養，理由見 2g（碰撞點）。")
res_rows, r2_rows, vifs = [], [], []


def add(res, terms, model, y, family=None):
    res_rows.append(tidy(res, terms, model=model, family=family or model, outcome=y))


for y in OUTC:
    lag = LAG[y]
    zb = [y] + GEN2 + Z2
    for v in DPV2:                                             # (a) 主模型
        add(fit(panel, y, [v] + GEN2 + COV2, zb + [v])[0], [v], "主模型", y)
    add(fit(panel, y, GEN2 + COV2, zb)[0], GEN2, "主模型", y)
    for v in DPV2 + GEN2:                                      # (b) 單獨放入，只控共變項
        add(fit(panel, y, [v] + COV2, [y, v] + Z2)[0], [v], "單一_僅共變項", y)
    for v in DPV2:                                             # (c) 主模型 + 控制國七同一結果（上床/睡眠僅 A 卷）
        add(fit(panel, y, [v] + GEN2 + COV2 + [lag], zb + [v, lag])[0], [v], "主模型+控制W1結果", y)
    add(fit(panel, y, GEN2 + COV2 + [lag], zb + [lag])[0], GEN2, "主模型+控制W1結果", y)
    # (d) 國七 B 卷教養 → 國八；成癮/憂鬱控國七同一結果＝前瞻；上床/睡眠 B 卷無國七作息題 → 只能控國七成癮+憂鬱
    lags, mname = (([lag], "W1前瞻(B卷,控W1結果)") if y not in SLEEPY
                   else (["addict_1", "cesd_nosleep_1"], "W1教養_未控W1作息(B卷)"))
    zb1 = [y] + GEN1 + Z1 + lags
    for v in DPV1:
        add(fit(panel, y, [v] + GEN1 + COV1 + lags, zb1 + [v])[0], [v], mname, y, "W1教養→W2(B卷)")
    add(fit(panel, y, GEN1 + COV1 + lags, zb1)[0], GEN1, mname, y, "W1教養→W2(B卷)")
    # (e) 敏感度：所有教養互相控制（可能有碰撞偏誤）；五構面版（鼓勵併入陪伴）；陪伴含鼓勵單獨
    r, d = fit(panel, y, DPV2 + GEN2 + COV2, zb + DPV2)
    add(r, DPV2 + GEN2, "敏感度_六構面互相控制", y)
    vifs += [1 / (1 - smf.ols(f"{v} ~ " + " + ".join(c for c in DPV2 + GEN2 if c != v), d).fit().rsquared)
             for v in DPV2 + GEN2]
    r0, _ = fit(d, y, COV2, [])                                # 增量 R²（同一樣本）
    r1, _ = fit(d, y, GEN2 + COV2, [])
    r2_rows.append({"outcome": y, "n": int(r.nobs), "R2_共變項": r0.rsquared, "R2_+一般親子": r1.rsquared,
                    "R2_+六種數位教養": r.rsquared, "ΔR2_數位教養(超越一般親子)": r.rsquared - r1.rsquared})
    add(fit(panel, y, DPV2_5 + GEN2 + COV2, zb + DPV2_5)[0], DPV2_5 + GEN2, "敏感度_五構面(鼓勵併入)互相控制", y)
    add(fit(panel, y, ["dp_active5_2"] + GEN2 + COV2, zb + ["dp_active5_2"])[0], ["dp_active5_2"],
        "敏感度_陪伴含鼓勵(單一+一般親子)", y)
coef = pd.concat(res_rows, ignore_index=True)
coef["dim"] = coef["term"].str.replace(r"^(dp_|par_)", "", regex=True).str[:-2]
coef = pd.concat([fdr(g) for _, g in coef.groupby("family", sort=False)])   # FDR：每個模型家族內全部教養係數
coef["sig"] = np.where(coef["q"] < .05, "*", "")
save(coef.round(4), "coef_main.csv", index=False)
fam_n = coef.groupby("family", sort=False).size()
log("BH-FDR 家族大小（每個模型家族內 4 結果 × 全部教養係數）：" + "、".join(f"{k} {v}" for k, v in fam_n.items()))
r2 = pd.DataFrame(r2_rows)
save(r2.round(4), "r2_increment.csv", index=False)
log(f"VIF（六種數位教養 + 一般親子同時放入時）：最大 {max(vifs):.2f}")


def show(model, outs=OUTC):
    t = coef[coef.model == model].copy()
    t["cell"] = t.apply(lambda r: f"{r.beta:+.3f} [{r.lo:+.3f},{r.hi:+.3f}]{'*' if r.q < .05 else ''}", axis=1)
    w = t.pivot(index="dim", columns="outcome", values="cell").reindex(columns=[c for c in outs if c in set(t.outcome)])
    w = w.reindex([k for k in DPLAB if k in w.index] + [k for k in w.index if k not in DPLAB])
    w.index = [DPLAB.get(k, k) for k in w.index]
    ns = t.groupby("outcome")["n"].agg(["min", "max"]).reindex(list(w.columns))
    w.columns = [outs[c] for c in w.columns]
    return w.to_string() + "\n  n = " + "；".join(f"{outs[o]} {a}–{b}" if a != b else f"{outs[o]} {a}"
                                                    for o, (a, b) in ns.iterrows())


log("### 2a. 主模型：國八同期，一次一種數位教養 + 一般親子 + 共變項（* = BH-FDR q<.05）\n" + show("主模型"))
log("\n### 2b. 單獨放入，只控共變項（最少調整）\n" + show("單一_僅共變項"))
log("\n### 2c. 主模型 + 控制國七同一結果（成癮/憂鬱全樣本；上床/睡眠僅 A 卷）。注意：教養仍是國八測量，"
    "這是「國七→國八殘差變化」的關聯，不是前瞻設計\n" + show("主模型+控制W1結果"))
log("\n### 2d. 國七 B 卷教養 → 國八。成癮/憂鬱：控國七同一結果（真正前瞻）；"
    "上床/睡眠：B 卷無國七作息題，只控國七成癮+憂鬱（＝未控基線作息，見 2e）\n"
    + show("W1前瞻(B卷,控W1結果)", {k: OUTC[k] for k in ("addict_2", "cesd_nosleep_2")}) + "\n"
    + show("W1教養_未控W1作息(B卷)", {k: OUTC[k] for k in SLEEPY}))

# 2e. A+B 卷同一樣本：控國七成癮+憂鬱（＝B 卷做法） vs 控國七作息 → 基線混淆有多大？
log("\n### 2e. 國七 B 卷「規則→早睡」是否被基線作息混淆？A+B 卷同一樣本比較兩種控制（學校叢集 bootstrap 差值）\n")
B_BOOT = 1000


def boot_change(d, y, v, xa, xb, reps=B_BOOT):
    """同一批列：v 的標準化係數（控制 xa）與（控制 xb）之差；學校叢集 bootstrap。"""
    Xa = patsy.dmatrix(" + ".join([v] + xa) + " + C(digi_level)", d, return_type="dataframe")
    Xb = patsy.dmatrix(" + ".join([v] + xb) + " + C(digi_level)", d, return_type="dataframe")
    ja, jb, Xa, Xb = Xa.columns.get_loc(v), Xb.columns.get_loc(v), Xa.values, Xb.values   # patsy 會把類別項排前面
    yv, xv, w = d[y].values, d[v].values, d["HOUWGT"].values
    sch = pd.factorize(d["nschool_id"])[0]
    pos = [np.flatnonzero(sch == s) for s in range(sch.max() + 1)]
    rg = np.random.default_rng(SEED)

    def beta(idx):
        sw, yy = np.sqrt(w[idx]), yv[idx]
        k = (wms(xv[idx], w[idx])[1] / wms(yy, w[idx])[1])
        ba = np.linalg.lstsq(Xa[idx] * sw[:, None], yy * sw, rcond=None)[0][ja] * k
        bb = np.linalg.lstsq(Xb[idx] * sw[:, None], yy * sw, rcond=None)[0][jb] * k
        return ba, bb
    full = beta(np.arange(len(d)))
    bs = np.array([np.subtract(*beta(np.concatenate([pos[s] for s in rg.integers(0, len(pos), len(pos))]))[::-1])
                   for _ in range(reps)])
    return full[0], full[1], full[1] - full[0], np.percentile(bs, [2.5, 97.5])


ab_rows = []
LAGAD = ["addict_1", "cesd_nosleep_1"]
for y in SLEEPY:
    lag = LAG[y]
    need = [y, lag] + LAGAD + DPV1 + GEN1 + COV1 + ["digi_level", "HOUWGT", "nschool_id"]
    dab = panel.dropna(subset=need)                            # 同一批列（所有設定共用）
    sp = [(v, "單一+一般親子", GEN1 + COV1) for v in DPV1] +          [("dp_restrict_1", "互相控制(敏感度)", [o for o in DPV1 if o != "dp_restrict_1"] + GEN1 + COV1)]
    for v, spec, base in sp:
        zc = [v] + [c for c in base if c not in ("female_1", "parents_married_cohab_1")]
        ra, _ = fit(dab, y, [v] + base + LAGAD, [y] + zc + LAGAD)
        rb, _ = fit(dab, y, [v] + base + [lag], [y] + zc + [lag])
        rw, _ = fit(dab, lag, [v] + base, [lag] + zc)          # 國七教養 ↔ 國七作息（基線混淆來源）
        ba, bb, chg, ci = boot_change(dab, y, v, base + LAGAD, base + [lag])
        assert abs(ba - ra.params[v]) < 1e-6 and abs(bb - rb.params[v]) < 1e-6
        ab_rows.append({"outcome": y, "教養": v[3:-2], "設定": spec, "n": int(ra.nobs), "schools": dab.nschool_id.nunique(),
                        "β_控W1成癮憂鬱": ba, "lo_a": ra.conf_int().loc[v, 0], "hi_a": ra.conf_int().loc[v, 1],
                        "β_控W1作息": bb, "lo_b": rb.conf_int().loc[v, 0], "hi_b": rb.conf_int().loc[v, 1],
                        "差值(b−a)": chg, "差值_lo": ci[0], "差值_hi": ci[1],
                        "W1教養→W1作息β": rw.params[v], "lo_w1": rw.conf_int().loc[v, 0], "hi_w1": rw.conf_int().loc[v, 1],
                        "p_w1": rw.pvalues[v]})
abt = pd.DataFrame(ab_rows)
save(abt.round(4), "w1_ab_baseline_check.csv", index=False)
log(abt.drop(columns=["schools", "lo_a", "hi_a", "lo_b", "hi_b"]).round(3).to_string(index=False))
log(f"（A+B 卷樣本 {abt.schools.max()} 校；bootstrap {B_BOOT} 次，重抽學校）")

# 2f. 時序負對照：國八規則「預測」一年前的國七上床時間 vs 同期國八上床時間（A 卷、同一樣本）
log("\n### 2f. 時序負對照：國八教養 → 國七作息（時間上不可能是效果） vs → 國八作息；A 卷同一樣本、主模型控制\n")
nc_rows = []
for y in SLEEPY:
    lag = LAG[y]
    for v in DPV2:
        dnc = panel.dropna(subset=[y, lag, v] + GEN2 + COV2 + ["digi_level", "HOUWGT", "nschool_id"])
        for tgt, lab in [(lag, "→國七(一年前)"), (y, "→國八(同期)")]:
            r, _ = fit(dnc, tgt, [v] + GEN2 + COV2, [tgt, v] + GEN2 + Z2)
            nc_rows.append(tidy(r, [v], outcome=y, 教養=v[3:-2], 目標=lab))
nct = pd.concat(nc_rows, ignore_index=True)
save(nct.round(4), "negative_control_w2rules_w1sleep.csv", index=False)
t = nct.assign(cell=nct.apply(lambda r: f"{r.beta:+.3f} [{r.lo:+.3f},{r.hi:+.3f}]", axis=1))
log(t.pivot_table(index="教養", columns=["outcome", "目標"], values="cell", aggfunc="first")
    .reindex(DPK).rename(index=DPLAB).to_string() + f"\n  n = {nct.groupby('outcome')['n'].first().to_dict()}")

# 2g. 碰撞點（collider）示範：規範限制的係數隨「控制了什麼」而變
log("\n### 2g. 規範限制的係數取決於調整集合（衝突責罵/擔憂可能是孩子問題上網的結果 → 碰撞點）\n")
ADJ = [("僅共變項", COV2), ("主模型：+一般親子", GEN2 + COV2),
       ("+一般親子+陪伴/鼓勵", ["dp_active_2", "dp_encourage_2"] + GEN2 + COV2),
       ("+一般親子+擔憂叮嚀", ["dp_worry_2"] + GEN2 + COV2), ("+一般親子+監看", ["dp_monitor_2"] + GEN2 + COV2),
       ("+一般親子+衝突責罵", ["dp_conflict_2"] + GEN2 + COV2),
       ("全部互相控制", [v for v in DPV2 if v != "dp_restrict_2"] + GEN2 + COV2)]
adj_rows = []
for y in OUTC:
    for lab, ctl in ADJ:
        r, _ = fit(panel, y, ["dp_restrict_2"] + ctl, [y, "dp_restrict_2"] + [c for c in ctl if c not in
                                                                             ("female_2", "parents_married_cohab_2")])
        adj_rows.append(tidy(r, ["dp_restrict_2"], outcome=y, 調整集合=lab))
adj = pd.concat(adj_rows, ignore_index=True)
save(adj.round(4), "restrict_adjustment_sets.csv", index=False)
t = adj.assign(cell=adj.apply(lambda r: f"{r.beta:+.3f} [{r.lo:+.3f},{r.hi:+.3f}]", axis=1))
log(t.pivot(index="調整集合", columns="outcome", values="cell").reindex([a for a, _ in ADJ])
    .reindex(columns=list(OUTC)).rename(columns=OUTC).to_string())
# 由加權相關直接分解：β(規則|衝突) = (r_ry − r_rc·r_cy) / (1 − r_rc²)
dcol = panel.dropna(subset=["dp_restrict_2", "dp_conflict_2", "HOUWGT"] + list(OUTC))
w_ = dcol["HOUWGT"]
r_rc = wcorr(dcol.dp_restrict_2, dcol.dp_conflict_2, w_)
col_rows = []
for y in OUTC:
    r_ry, r_cy = wcorr(dcol.dp_restrict_2, dcol[y], w_), wcorr(dcol.dp_conflict_2, dcol[y], w_)
    col_rows.append({"outcome": y, "n": len(dcol), "r(規則,Y)": r_ry, "r(規則,衝突)": r_rc, "r(衝突,Y)": r_cy,
                     "β規則|衝突": (r_ry - r_rc * r_cy) / (1 - r_rc ** 2),
                     "來自規則邊際相關": r_ry / (1 - r_rc ** 2), "來自衝突路徑": -r_rc * r_cy / (1 - r_rc ** 2)})
colt = pd.DataFrame(col_rows)
save(colt.round(4), "collider_decomposition.csv", index=False)
log("\n以加權相關分解（無共變項；同一樣本）：\n" + colt.round(3).to_string(index=False))
log("解讀：β規則|衝突 = [r(規則,Y) − r(規則,衝突)·r(衝突,Y)] / [1 − r(規則,衝突)²]。成癮/憂鬱上，規則的邊際相關≈0，"
    "控制衝突後出現的負係數全部來自第二項（衝突路徑）。"
    "若衝突是「規則」與「孩子問題上網」的共同結果（碰撞點），這是偏誤；若衝突是規則的中介，則負係數只是「固定衝突時」的直接關聯。"
    "兩種情況都不支持「多立規則→較少成癮/憂鬱」的總效果。上床/睡眠則不論怎麼調整都穩定。")

# 2h. 規範限制 × 衝突責罵（敏感度；以碰撞點為條件）
log("\n### 2h. 敏感度：規範限制 × 衝突責罵（以衝突為條件，可能有碰撞偏誤，僅供參考；BH-FDR 4 項內）\n")
rc_rows = []
for y in OUTC:
    r, _ = fit(panel, y, ["dp_restrict_2", "dp_conflict_2"] + GEN2 + COV2,
               [y, "dp_restrict_2", "dp_conflict_2"] + GEN2 + Z2, extra=" + dp_restrict_2:dp_conflict_2")
    ti = "dp_restrict_2:dp_conflict_2"
    row = {"outcome": y, "n": int(r.nobs), "beta_restrict": r.params["dp_restrict_2"], "beta_int": r.params[ti],
           "lo": r.conf_int().loc[ti, 0], "hi": r.conf_int().loc[ti, 1], "p": r.pvalues[ti]}
    for lv in (-1, 1):
        row[f"restrict斜率_衝突{lv:+d}SD"] = r.params["dp_restrict_2"] + lv * r.params[ti]
    rc_rows.append(row)
rc = fdr(pd.DataFrame(rc_rows))
save(rc.round(4), "restrict_x_conflict.csv", index=False)
log(rc.round(3).to_string(index=False))

log("\n### 2i. 敏感度：互相控制模型（六構面；五構面＝鼓勵併入陪伴）\n" + show("敏感度_六構面互相控制")
    + "\n\n五構面（EFA 版，鼓勵併入陪伴）：\n" + show("敏感度_五構面(鼓勵併入)互相控制")
    + "\n\n陪伴(含鼓勵) 單一 + 一般親子：\n" + show("敏感度_陪伴含鼓勵(單一+一般親子)"))
log("\n增量 R²（同一樣本；六種數位教養一起加入）：\n" + r2.round(3).to_string(index=False))

# 四格描述：規則有/少 × 責罵有/幾乎沒有（加權平均 ± 叢集 95% CI；未調整）
RH, CH = 3, 2                                                  # 規則：父母平均 ≥「有時」；責罵：≥「偶爾」
GORD = ["有規則・幾乎不責罵", "少規則・幾乎不責罵", "有規則・有責罵", "少規則・有責罵"]
g4 = panel.dropna(subset=["dp_restrict_2", "dp_conflict_2", "HOUWGT"]).copy()
g4["grp"] = np.select([(g4.dp_restrict_2 >= RH) & (g4.dp_conflict_2 < CH), g4.dp_restrict_2 >= RH,
                       g4.dp_conflict_2 < CH], [GORD[0], GORD[2], GORD[1]], GORD[3])
g4["late"] = (g4["bedtime_2"] >= 24).astype(float).where(g4["bedtime_2"].notna()) * 100
grows = []
for v, vl in [("late", "午夜(含)後才睡 %"), ("addict_2", "網路成癮 (1–4)"), ("cesd_nosleep_2", "憂鬱 CESD（不含睡眠題）")]:
    dd = g4.dropna(subset=[v])
    r = smf.wls(f"{v} ~ 0 + C(grp)", dd, weights=dd["HOUWGT"]).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(dd["nschool_id"])[0]})
    for gname in GORD:
        t_ = f"C(grp)[{gname}]"
        grows.append({"變項": vl, "組別": gname, "n": int((dd.grp == gname).sum()), "mean": r.params[t_],
                      "lo": r.conf_int().loc[t_, 0], "hi": r.conf_int().loc[t_, 1]})
grp = pd.DataFrame(grows)
save(grp.round(3), "restrict_conflict_groups.csv", index=False)
gsh = g4.grp.value_counts(normalize=True).reindex(GORD) * 100
log(f"\n四格描述（規則 ≥{RH}「有時」、責罵 ≥{CH}「偶爾」；加權、未調整）。各組占比："
    + "、".join(f"{g} {s:.1f}%" for g, s in gsh.items()) + "\n" + grp.round(2).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.1))
GCOL = {GORD[0]: BLUE, GORD[1]: "#86b6ef", GORD[2]: ORANGE, GORD[3]: "#f4b49a"}
FMT = {"午夜(含)後才睡 %": "{:.1f}%", "網路成癮 (1–4)": "{:.2f}", "憂鬱 CESD（不含睡眠題）": "{:.1f}"}
for ax, (vl, t_) in zip(axes, grp.groupby("變項", sort=False)):
    t_ = t_.set_index("組別").reindex(GORD)
    for i, g in enumerate(GORD):
        ax.errorbar(i, t_.at[g, "mean"], yerr=[[t_.at[g, "mean"] - t_.at[g, "lo"]], [t_.at[g, "hi"] - t_.at[g, "mean"]]],
                    fmt="o", ms=9, color=GCOL[g], mec="white", mew=1.5, elinewidth=2, capsize=0, zorder=3)
        ax.text(i + .13, t_.at[g, "mean"], FMT[vl].format(t_.at[g, "mean"]), va="center", fontsize=9, color=INK)
    ax.axvline(1.5, color="#d6d5d0", lw=1, ls="--")
    ax.set_xticks(range(4), [g.replace("・", "\n") for g in GORD], fontsize=8.5)
    ax.set_xlim(-.5, 3.7)
    lo_, hi_ = t_["lo"].min(), t_["hi"].max()
    ax.set_ylim(lo_ - .25 * (hi_ - lo_), hi_ + .15 * (hi_ - lo_))
    ax.set_title(vl, fontsize=11)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(axis="y", color="#ecebe6", lw=.8)
    ax.set_axisbelow(True)
ns_ = grp.drop_duplicates("組別").set_index("組別")["n"].reindex(GORD)
fig.suptitle("成癮與憂鬱的落差主要跟著「責罵」走；規則主要與較早睡有關（加權平均 ± 95% CI，未調整）", fontsize=12)
fig.text(.5, .085, "規則＝父母「規定時段/限制時長」平均達「有時」以上；有責罵＝「因上網責罵/網購衝突」平均達「偶爾」以上（約三分之一學生）。",
         ha="center", fontsize=8, color=MUTED)
fig.text(.5, .05, "n（上床）：" + "、".join(f"{g} {n:,}" for g, n in ns_.items()), ha="center", fontsize=8, color=MUTED)
fig.text(.5, .015, "注意：責罵可能是孩子問題上網的結果；在同一責罵層內比較有無規則，等於以碰撞點為條件，僅作描述，不代表規則的效果。",
         ha="center", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, .11, 1, .93))
fig.savefig(FIG / "fig4_rules_vs_scolding.png", dpi=200)
plt.close(fig)

# 圖 1：教養 × 結果 係數圖（主模型）
ORDER = DPK + ["warmth", "hostile", "cohesion"]
XLAB = {"bedtime_2": "← 較早睡　　較晚睡 →", "sleep_hrs_2": "← 睡較少　　睡較多 →",
        "addict_2": "← 較少成癮　　較多 →", "cesd_nosleep_2": "← 較不憂鬱　　較憂鬱 →"}
fig, axes = plt.subplots(1, 4, figsize=(13.2, 5.8), sharey=True)
ser = [("主模型", "國八同期（全樣本）", BLUE, "o", -.17), ("主模型+控制W1結果", "再控制國七同一結果", ORANGE, "s", .17)]
for ax, y in zip(axes, OUTC):
    for model, lab, colr, mk, off in ser:
        t = coef[(coef.model == model) & (coef.outcome == y)].set_index("dim").reindex(ORDER)
        ypos = np.arange(len(ORDER)) + off
        ok = t["beta"].notna().values
        ax.errorbar(t["beta"][ok], ypos[ok], xerr=[(t.beta - t.lo)[ok], (t.hi - t.beta)[ok]], fmt="none",
                    ecolor=colr, elinewidth=1.6, capsize=0)
        sig = (t["q"] < .05).values & ok
        ax.scatter(t["beta"][sig], ypos[sig], marker=mk, s=42, color=colr, zorder=3, label=lab)
        ax.scatter(t["beta"][~sig & ok], ypos[~sig & ok], marker=mk, s=42, facecolor="white", edgecolor=colr,
                   lw=1.4, zorder=3)
    ax.axvline(0, color=GRAY, lw=1)
    ax.axhline(len(DPK) - .5, color="#d6d5d0", lw=1, ls="--")
    ax.set_title(OUTC[y], fontsize=12)
    ax.set_xlabel(XLAB[y], fontsize=9, color=MUTED)
    ax.grid(axis="x", color="#ecebe6", lw=.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    n1 = int(coef[(coef.model == "主模型") & (coef.outcome == y) & (coef.dim == "restrict")]["n"].iloc[0])
    n2 = int(coef[(coef.model == "主模型+控制W1結果") & (coef.outcome == y) & (coef.dim == "restrict")]["n"].iloc[0])
    ax.text(.98, .01, f"n≈{n1:,} / {n2:,}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED)
axes[0].set_yticks(range(len(ORDER)), [DPLAB[k] for k in ORDER], fontsize=10.5)
axes[0].invert_yaxis()
axes[0].text(-.52, (len(DPK) - 1) / 2, "數位教養", transform=axes[0].get_yaxis_transform(), rotation=90, va="center",
             ha="center", fontsize=10, color=MUTED)
axes[0].text(-.52, len(DPK) + 1, "一般親子", transform=axes[0].get_yaxis_transform(), rotation=90, va="center",
             ha="center", fontsize=10, color=MUTED)
h1 = plt.Line2D([], [], color=BLUE, marker="o", lw=1.6, label="國八同期（全樣本）")
h2 = plt.Line2D([], [], color=ORANGE, marker="s", lw=1.6, label="再控制國七同一結果")
h3 = plt.Line2D([], [], color=GRAY, marker="o", lw=0, markerfacecolor="white", label="空心 = FDR q ≥ .05")
fig.legend(handles=[h1, h2, h3], loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(.5, .045))
fig.text(.5, .025, "每種數位教養各自一個模型（該教養 + 溫暖/敵意/凝聚 + 性別、主觀社經、父母婚姻、數位發展區），不互相控制；一般親子三列取自「一般親子三項 + 共變項」模型。",
         ha="center", fontsize=8, color=MUTED)
fig.text(.5, .005, "n = 同期 / 再控制國七；再控制國七：成癮、憂鬱為全樣本，上床、睡眠僅國七 A 卷。加權（HOUWGT），學校叢集標準誤，FDR 為每種模型 36 項內校正。",
         ha="center", fontsize=8, color=MUTED)
fig.suptitle("父母怎麼「管網路」與國八學生的作息、成癮、憂鬱（標準化 β ± 95% CI）", fontsize=12.5, y=.985)
fig.tight_layout(rect=(0.02, .1, 1, .96))
fig.savefig(FIG / "fig1_parenting_coef.png", dpi=200)
plt.close(fig)

# 圖 S2：規範限制的係數 × 調整集合
fig, axes = plt.subplots(1, 4, figsize=(13.2, 4.3), sharey=True)
SUSP = {"+一般親子+擔憂叮嚀", "+一般親子+監看", "+一般親子+衝突責罵", "全部互相控制"}
for ax, y in zip(axes, OUTC):
    t = adj[adj.outcome == y].set_index("調整集合").reindex([a for a, _ in ADJ])
    for i, (a, r_) in enumerate(t.iterrows()):
        colr = BLUE if a.startswith("主模型") else (ORANGE if a in SUSP else GRAY)
        ax.errorbar(r_.beta, i, xerr=[[r_.beta - r_.lo], [r_.hi - r_.beta]], fmt="o", color=colr, ms=6,
                    elinewidth=1.6, capsize=0, mec="white", mew=1)
    ax.axvline(0, color=GRAY, lw=1)
    ax.set_title(OUTC[y], fontsize=11.5)
    ax.set_xlabel("規範限制的標準化 β（95% CI）", fontsize=9, color=MUTED)
    ax.grid(axis="x", color="#ecebe6", lw=.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_yticks(range(len(ADJ)), [a for a, _ in ADJ], fontsize=9.5)
axes[0].invert_yaxis()
h1 = plt.Line2D([], [], color=BLUE, marker="o", lw=0, label="主模型")
h2 = plt.Line2D([], [], color=ORANGE, marker="o", lw=0, label="控制了可能是孩子狀況「結果」的教養（碰撞點疑慮）")
h3 = plt.Line2D([], [], color=GRAY, marker="o", lw=0, label="其他")
fig.legend(handles=[h1, h2, h3], loc="lower center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(.5, -.01))
fig.suptitle("「規則」的好處取決於控制了什麼：上床/睡眠一路穩定；成癮/憂鬱只在控制責罵、擔憂等教養後才「出現」", fontsize=12)
fig.tight_layout(rect=(0, .07, 1, .95))
fig.savefig(FIG / "fig_s2_restrict_adjustment.png", dpi=200)
plt.close(fig)

# 圖 S3：碰撞點示意（DAG）
cr = colt.set_index("outcome").loc["addict_2"]
fig, ax = plt.subplots(figsize=(7.6, 5.0))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
NODE = {"bg": (.5, .9, "家庭背景・一般親子關係", "#ecebe6"), "R": (.15, .58, "規範限制\n（規定時段、限制時長）", "#dbe8f9"),
        "Y": (.85, .58, "孩子網路成癮／憂鬱", "#fde4d9"), "C": (.5, .25, "衝突責罵\n（因上網責罵、網購衝突）", "#f3f2ee")}
for x_, y_, txt, fc in NODE.values():
    ax.add_patch(FancyBboxPatch((x_ - .14, y_ - .065), .28, .13, boxstyle="round,pad=.01,rounding_size=.02",
                                fc=fc, ec="#bdbcb6", lw=1))
    ax.text(x_, y_, txt, ha="center", va="center", fontsize=10, color=INK)


def arrow(p0, p1, colr, ls="-", lw=1.6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16, color=colr, lw=lw, ls=ls,
                                 shrinkA=2, shrinkB=2))


arrow((.40, .83), (.22, .65), GRAY)                             # 端點設在方框邊緣（方框寬 .28、高 .13）
arrow((.60, .83), (.78, .65), GRAY)
arrow((.29, .58), (.71, .58), BLUE, ls="--", lw=2)
arrow((.22, .51), (.40, .32), INK)
arrow((.78, .51), (.60, .32), ORANGE, lw=2)
ax.text(.5, .62, "？想估計的效果", ha="center", fontsize=9.5, color=BLUE)
ax.text(.24, .36, "執行規則\n引發衝突", ha="center", fontsize=9, color=INK)
ax.text(.77, .36, "因孩子上網過度\n而責罵（子→親）", ha="center", fontsize=9, color=ORANGE)
ax.text(.5, .74, "混淆\n（以一般親子關係等控制）", ha="center", va="center", fontsize=8.5, color=MUTED)
ax.text(.5, .05, f"控制「衝突責罵」＝以碰撞點為條件：即使規則對孩子毫無影響，也會讓規則與成癮呈負相關。\n"
        f"國八資料：r(規則,成癮)={cr['r(規則,Y)']:+.2f}，r(規則,衝突)={cr['r(規則,衝突)']:.2f}，r(衝突,成癮)={cr['r(衝突,Y)']:.2f}"
        f" → 控制衝突後 β≈{cr['β規則|衝突']:+.2f}（邊際 {cr['來自規則邊際相關']:+.2f}、衝突路徑 {cr['來自衝突路徑']:+.2f}）",
        ha="center", va="center", fontsize=8.5, color=INK)
ax.set_title("為什麼主模型不同時控制「責罵」？（碰撞點示意）", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "fig_s3_collider_dag.png", dpi=200)
plt.close(fig)

# ---------- 3. 調節：成癮/上床時間 × 教養 → 憂鬱 ----------
log("\n## 3. 調節（緩衝）分析：結果 = 國八憂鬱（不含睡眠題）；X、M 皆加權標準化（已中心化）；"
    "每式只放一種教養 M（+ 一般親子 + 共變項），與主模型一致\n")
MODS = DPK + ["warmth"]
int_rows, fits, specs = [], {}, []
for x in ("addict_2", "bedtime_2"):
    for k in MODS:
        m = f"dp_{k}_2" if k in DPK else "warmth_2"
        specs.append(("同期", x, m, [x, m] + GEN2 + COV2, [x, m] + GEN2 + Z2))
for k in MODS:
    m = f"dp_{k}_2" if k in DPK else "warmth_2"
    specs.append(("半前瞻(W1成癮×W2教養)", "addict_1", m, ["addict_1", m, "cesd_nosleep_1"] + GEN2 + COV2,
                  ["addict_1", m, "cesd_nosleep_1"] + GEN2 + Z2))
for k in MODS:
    m = f"dp_{k}_1" if k in DPK else "warmth_1"
    specs.append(("W1前瞻(B卷)", "addict_1", m, ["addict_1", m, "cesd_nosleep_1"] + GEN1 + COV1,
                  ["addict_1", m, "cesd_nosleep_1"] + GEN1 + Z1))
y = "cesd_nosleep_2"
for kind, x, m, xs, zc in specs:
    r, d = fit(panel, y, xs, [y] + zc, extra=f" + {x}:{m}")
    ti = f"{x}:{m}"
    V = r.cov_params()
    ss = {}
    for lv in (-1, 0, 1):
        b = r.params[x] + lv * r.params[ti]
        se = np.sqrt(V.loc[x, x] + lv ** 2 * V.loc[ti, ti] + 2 * lv * V.loc[x, ti])
        ss[lv] = (b, se)
    int_rows.append({"類型": kind, "X": x, "M": m, "n": int(r.nobs), "beta_X": r.params[x], "beta_int": r.params[ti],
                     "lo": r.conf_int().loc[ti, 0], "hi": r.conf_int().loc[ti, 1], "p": r.pvalues[ti],
                     "slope_M-1SD": ss[-1][0], "se_M-1SD": ss[-1][1], "slope_M+1SD": ss[1][0], "se_M+1SD": ss[1][1]})
    fits[(kind, x, m)] = (r, d)
inter = fdr(pd.DataFrame(int_rows))
save(inter.round(4), "interactions.csv", index=False)
show_i = inter.copy()
show_i["M"] = show_i["M"].str.replace(r"^(dp_)", "", regex=True).str[:-2].map(DPLAB)
log(show_i[["類型", "X", "M", "n", "beta_X", "beta_int", "lo", "hi", "p", "q", "slope_M-1SD", "slope_M+1SD"]]
    .round(3).to_string(index=False))
log(f"（BH-FDR 跨全部 {len(inter)} 個交互作用項校正）")

# 最明確的交互作用 → 簡單斜率 + Johnson–Neyman
best = inter.sort_values("p").iloc[0]
key = (best["類型"], best["X"], best["M"])
r, d = fits[key]
x, m = best["X"], best["M"]
ti = f"{x}:{m}"
raw_x, raw_m, raw_y = panel.loc[d.index, x], panel.loc[d.index, m], panel.loc[d.index, y]
w = d["HOUWGT"]
(mx, sx), (mm, sm), (my, sy) = wms(raw_x, w), wms(raw_m, w), wms(raw_y, w)
exog = pd.DataFrame(r.model.exog, columns=r.model.exog_names)
base = pd.Series(np.average(exog, axis=0, weights=w), index=exog.columns)
V = r.cov_params()
XLABS = {"addict_2": "國八網路成癮", "bedtime_2": "國八上床時間", "addict_1": "國七網路成癮"}
mk = m.replace("dp_", "")[:-2]
mname = DPLAB[mk]
log(f"\n最明確交互作用：{best['類型']} {XLABS[x]} × {mname}（β_int={best['beta_int']:+.3f}, "
    f"95% CI [{best['lo']:+.3f}, {best['hi']:+.3f}], {pfmt(best['p'])}, {qfmt(best['q'])}, n={best['n']}）")

fig = plt.figure(figsize=(11, 4.9))
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[4, 1], hspace=.08)
a1, a2 = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1])
a3 = fig.add_subplot(gs[1, 1], sharex=a2)
xs_z = np.linspace(-1.5, 2.5, 60)
for lv, colr, lab in [(-1, "#86b6ef", f"{mname}低（−1 SD）"), (0, BLUE, "平均"), (1, "#104281", f"{mname}高（+1 SD）")]:
    G = np.tile(base.values, (len(xs_z), 1))
    G[:, exog.columns.get_loc(x)] = xs_z
    G[:, exog.columns.get_loc(m)] = lv
    G[:, exog.columns.get_loc(ti)] = xs_z * lv
    pred = G @ r.params.values
    se = np.sqrt(np.einsum("ij,jk,ik->i", G, V.values, G))
    xr = mx + xs_z * sx
    a1.plot(xr, my + pred * sy, color=colr, lw=2.2, label=lab)
    a1.fill_between(xr, my + (pred - 1.96 * se) * sy, my + (pred + 1.96 * se) * sy, color=colr, alpha=.15, lw=0)
    b = r.params[x] + lv * r.params[ti]
    a1.text(xr[-1] + .03 * (xr[-1] - xr[0]), my + pred[-1] * sy, f"β={b:+.2f}", color=INK, va="center", fontsize=8.5)
a1.set_xlabel(f"{XLABS[x]}（原始分數）")
a1.set_ylabel("國八憂鬱（CESD 不含睡眠題，0–44）")
a1.set_title("簡單斜率：預測憂鬱分數（95% CI）", fontsize=11)
a1.legend(frameon=False, fontsize=9, loc="upper left")
a1.set_xlim(xr[0], xr[-1] + .2 * (xr[-1] - xr[0]))
mz = np.linspace((raw_m.min() - mm) / sm, (raw_m.max() - mm) / sm, 300)
sl = r.params[x] + mz * r.params[ti]
se = np.sqrt(V.loc[x, x] + mz ** 2 * V.loc[ti, ti] + 2 * mz * V.loc[x, ti])
mr = mm + mz * sm
a2.plot(mr, sl, color=BLUE, lw=2.2)
a2.fill_between(mr, sl - 1.96 * se, sl + 1.96 * se, color=BLUE, alpha=.18, lw=0)
a2.axhline(0, color=GRAY, lw=1)
sig_jn = (sl - 1.96 * se > 0) | (sl + 1.96 * se < 0)
cut = np.where(np.diff(sig_jn.astype(int)) != 0)[0]
for i in cut:
    jn = (mr[i] + mr[i + 1]) / 2
    a2.axvline(jn, color=ORANGE, ls="--", lw=1.4)
    a2.text(jn, .97, f" J–N ≈ {jn:.2f}", color=ORANGE, va="top", fontsize=9, transform=a2.get_xaxis_transform())
    side = "以上" if sig_jn[i + 1] else "以下"
    log(f"Johnson–Neyman：{mname} 約 {jn:.2f}（原始 1–4 分）{side}時，{XLABS[x]}對憂鬱的斜率顯著不為 0")
if not len(cut):
    log(f"Johnson–Neyman：在觀察範圍內斜率{'全程顯著' if sig_jn.all() else '全程不顯著'}（無臨界點）；"
        f"斜率範圍 {sl.min():+.3f} 至 {sl.max():+.3f}")
a2.set_xlim(mr[0], mr[-1])
a2.set_ylim(min(0, (sl - 1.96 * se).min()) - .02, (sl + 1.96 * se).max() + .03)
a2.set_ylabel("成癮→憂鬱斜率（標準化 β）")
a2.set_title("Johnson–Neyman：斜率隨調節變項變化（95% CI）", fontsize=11)
a2.tick_params(labelbottom=False)
a2.grid(axis="y", color="#ecebe6", lw=.8)
if not len(cut):
    a2.text(.03, .14, "全程 CI 不含 0：沒有任何程度能讓關聯消失", transform=a2.transAxes, fontsize=8.5, color=MUTED)
a3.hist(raw_m, bins=np.arange(.875, 4.2, .25), weights=w / w.sum() * 100, color="#bdbcb6", edgecolor="white")
a3.set_ylabel("%", fontsize=9)
a3.set_xlabel(f"{mname}（父母平均，1 從未 – 4 經常）")
for a in (a1, a2, a3):
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
lab = {"同期": "國八同期", "半前瞻(W1成癮×W2教養)": "國七成癮 × 國八教養 → 國八憂鬱（控國七憂鬱）",
       "W1前瞻(B卷)": "國七 B 卷：國七成癮 × 國七教養 → 國八憂鬱（控國七憂鬱）"}[best["類型"]]
fig.suptitle(f"{XLABS[x]} × {mname} → 憂鬱（{lab}；交互作用 β={best['beta_int']:+.3f}，{qfmt(best['q'])}，n={best['n']:,}）",
             fontsize=11.5)
fig.subplots_adjust(left=.07, right=.98, top=.86, bottom=.12, wspace=.25)
fig.savefig(FIG / "fig2_interaction_simple_slopes.png", dpi=200)
plt.close(fig)

# ---------- 4. 反向因果：W1→W2 交叉延宕（B 卷） ----------
log("\n## 4. 反向因果檢查：國七→國八交叉延宕（B 卷；標準化 β；各式控制性別、家庭經濟、父母婚姻、數位發展區）\n")
log("註：每組只含一種數位教養與一個孩子結果（單一構面模型），未控制其他教養或一般親子；"
    "兩波資料無法估計隨機截距（RI-CLPM），係數混合了家庭間穩定差異與家庭內變化。")
cl_rows, df_rows = [], []
for k in DPK:
    for yo, ylab in [("addict", "網路成癮"), ("cesd_nosleep", "憂鬱")]:
        p1, p2, c1, c2 = f"dp_{k}_1", f"dp_{k}_2", f"{yo}_1", f"{yo}_2"
        d0 = panel.dropna(subset=[p1, p2, c1, c2] + COV1 + ["digi_level", "HOUWGT", "nschool_id"])
        ra, _ = fit(d0, c2, [c1, p1] + COV1, [c2, c1, p1] + Z1)
        cl_rows.append(tidy(ra, [p1], 方向="親→子", 教養=k, 結果=ylab))
        rb, _ = fit(d0, p2, [p1, c1] + COV1, [p2, p1, c1] + Z1)
        cl_rows.append(tidy(rb, [c1], 方向="子→親", 教養=k, 結果=ylab))
        # 兩條方程式堆疊（同一列、同一學校叢集）→ 叢集穩健共變異含跨式共變 → 檢定 β親→子 = β子→親
        dz = d0.copy()
        for c in [p1, p2, c1, c2] + Z1:
            dz[c] = wz(dz[c], dz["HOUWGT"])
        X = patsy.dmatrix(" + ".join([c1, p1] + COV1) + " + C(digi_level)", dz, return_type="dataframe")
        kx, O = X.shape[1], np.zeros(X.shape)
        st_ = smapi.WLS(np.r_[dz[c2], dz[p2]], np.block([[X.values, O], [O, X.values]]),
                     weights=np.r_[dz.HOUWGT, dz.HOUWGT]).fit(
            cov_type="cluster", cov_kwds={"groups": np.tile(pd.factorize(dz.nschool_id)[0], 2)})
        ia, ib = X.columns.get_loc(p1), kx + X.columns.get_loc(c1)
        assert abs(st_.params[ia] - ra.params[p1]) < 1e-8 and abs(st_.params[ib] - rb.params[c1]) < 1e-8
        L_ = np.zeros(2 * kx)
        L_[ia], L_[ib] = 1, -1
        dv, dse = L_ @ st_.params, np.sqrt(L_ @ st_.cov_params() @ L_)
        df_rows.append({"教養": k, "結果": ylab, "n": int(ra.nobs), "β親→子": ra.params[p1], "β子→親": rb.params[c1],
                        "差(親→子−子→親)": dv, "lo": dv - 1.96 * dse, "hi": dv + 1.96 * dse,
                        "z": dv / dse, "p": 2 * stats.norm.sf(abs(dv / dse))})
clpm = fdr(pd.concat(cl_rows, ignore_index=True))
save(clpm.round(4), "cross_lagged.csv", index=False)
cdiff = fdr(pd.DataFrame(df_rows))
save(cdiff.round(4), "cross_lagged_path_difference.csv", index=False)
t = clpm.copy()
t["cell"] = t.apply(lambda r: f"{r.beta:+.3f} [{r.lo:+.3f},{r.hi:+.3f}] q={r.q:.3f}", axis=1)
t["教養"] = t["教養"].map(DPLAB)
log(t.pivot_table(index="教養", columns=["結果", "方向"], values="cell", aggfunc="first").reindex(
    [DPLAB[k] for k in DPK]).to_string())
log(f"n 範圍：{clpm['n'].min()}–{clpm['n'].max()}（BH-FDR 跨 {len(clpm)} 條路徑）")
log("\n兩方向是否不同（堆疊方程式、學校叢集穩健 Wald 檢定；BH-FDR 跨 12 對）：\n"
    + cdiff.assign(教養=cdiff["教養"].map(DPLAB)).round(3).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
for ax, yl in zip(axes, ["網路成癮", "憂鬱"]):
    for dirn, colr, mk_, off in [("親→子", BLUE, "o", -.15), ("子→親", ORANGE, "s", .15)]:
        tt = clpm[(clpm["結果"] == yl) & (clpm["方向"] == dirn)].set_index("教養").reindex(DPK)
        yy = np.arange(len(DPK)) + off
        ax.errorbar(tt.beta, yy, xerr=[tt.beta - tt.lo, tt.hi - tt.beta], fmt="none", ecolor=colr, elinewidth=1.6)
        sig = (tt.q < .05).values
        ax.scatter(tt.beta[sig], yy[sig], marker=mk_, s=42, color=colr, zorder=3)
        ax.scatter(tt.beta[~sig], yy[~sig], marker=mk_, s=42, facecolor="white", edgecolor=colr, lw=1.4, zorder=3)
    dd_ = cdiff[cdiff["結果"] == yl].set_index("教養").reindex(DPK)
    for i, k in enumerate(DPK):
        ax.text(1.0, i, f"差異 p={dd_.at[k, 'p']:.3f}" if dd_.at[k, "p"] >= .001 else "差異 p<.001",
                transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.5,
                color=INK if dd_.at[k, "q"] < .05 else GRAY)
    ax.axvline(0, color=GRAY, lw=1)
    ax.set_title(f"數位教養 ⇄ {yl}", fontsize=11.5)
    ax.set_xlabel("標準化交叉延宕係數（95% CI）", fontsize=9)
    ax.grid(axis="x", color="#ecebe6", lw=.8)
    ax.set_axisbelow(True)
    lo_, hi_ = clpm[clpm["結果"] == yl][["lo", "hi"]].values.min(), clpm[clpm["結果"] == yl][["lo", "hi"]].values.max()
    ax.set_xlim(lo_ - .01, hi_ + .07)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_yticks(range(len(DPK)), [DPLAB[k] for k in DPK])
axes[0].invert_yaxis()
h1 = plt.Line2D([], [], color=BLUE, marker="o", lw=1.6, label="親→子：國七教養 → 國八孩子（控國七孩子）")
h2 = plt.Line2D([], [], color=ORANGE, marker="s", lw=1.6, label="子→親：國七孩子 → 國八教養（控國七教養）")
h3 = plt.Line2D([], [], color=GRAY, marker="o", lw=0, markerfacecolor="white", label="空心 = FDR q ≥ .05")
fig.legend(handles=[h1, h2, h3], loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(.5, .03))
fig.text(.5, .005, "右側「差異 p」：兩方向係數是否相等（堆疊方程式、學校叢集 Wald 檢定；黑字 = FDR q<.05）。單一構面模型，兩波無法分離家庭間穩定差異。",
         ha="center", fontsize=7.8, color=MUTED)
fig.suptitle(f"是父母管法影響孩子，還是孩子狀況改變父母管法？（國七 B 卷交叉延宕，n≈{int(clpm['n'].median()):,}）",
             fontsize=12)
fig.tight_layout(rect=(0, .13, 1, .94))
fig.savefig(FIG / "fig3_cross_lagged.png", dpi=200)
plt.close(fig)


# ---------- 5. 重點摘要（數字自動帶入） ----------
def cb(model, y, dim, short=False):
    r_ = coef[(coef.model == model) & (coef.outcome == y) & (coef.dim == dim)].iloc[0]
    q_ = "q<.001" if r_.q < .001 else f"q={r_.q:.3f}"
    return f"β={r_.beta:+.3f} [{r_.lo:+.3f}, {r_.hi:+.3f}]" + ("" if short else f", {pfmt(r_.p)}, {q_}, n={r_.n:,}")


def cl(dim, yl, dirn):
    r_ = clpm[(clpm["教養"] == dim) & (clpm["結果"] == yl) & (clpm["方向"] == dirn)].iloc[0]
    return f"β={r_.beta:+.3f} [{r_.lo:+.3f}, {r_.hi:+.3f}], {pfmt(r_.p)}, q={r_.q:.3f}, n={r_.n:,}"


def cd(dim, yl):
    r_ = cdiff[(cdiff["教養"] == dim) & (cdiff["結果"] == yl)].iloc[0]
    return f"差={r_['差(親→子−子→親)']:+.3f} [{r_.lo:+.3f}, {r_.hi:+.3f}], {pfmt(r_.p)}"


def ab(y, spec="單一+一般親子", k="restrict"):
    return abt[(abt.outcome == y) & (abt["教養"] == k) & (abt["設定"] == spec)].iloc[0]


def abf(r_):
    return (f"{r_['β_控W1成癮憂鬱']:+.3f} → {r_['β_控W1作息']:+.3f}（差 {r_['差值(b−a)']:+.3f}，bootstrap 95% CI "
            f"[{r_.差值_lo:+.3f}, {r_.差值_hi:+.3f}]）")


def nc(y, tgt, k="restrict"):
    return nct[(nct.outcome == y) & (nct["教養"] == k) & (nct["目標"] == tgt)].iloc[0]


def ncf(y, tgt):
    r_ = nc(y, tgt)
    return f"β={r_.beta:+.3f} [{r_.lo:+.3f}, {r_.hi:+.3f}]"


def ad(y, a):
    r_ = adj[(adj.outcome == y) & (adj["調整集合"] == a)].iloc[0]
    return f"β={r_.beta:+.3f} [{r_.lo:+.3f}, {r_.hi:+.3f}]"


pint = inter[inter["類型"] != "同期"]
rca = rc.set_index("outcome").loc["addict_2"]
a_mon = alpha[(alpha["版本"].str.startswith("W2")) & (alpha["構面"] == "monitor")]["α"].iloc[0]
r_mc = lat[(lat.lval == "CON") & (lat.rval == "MON")]["r"].iloc[0]
isame = inter[(inter["類型"] == "同期") & (inter.q < .05)]
ca, cc_ = colt.set_index("outcome").loc["addict_2"], colt.set_index("outcome").loc["cesd_nosleep_2"]
db = panel.dropna(subset=["bedtime_2", "HOUWGT"])
sd_bed = wms(db.bedtime_2, db.HOUWGT)[1]
b_rb = coef[(coef.model == "主模型") & (coef.outcome == "bedtime_2") & (coef.dim == "restrict")].beta.iloc[0]
log("\n## 5. 重點摘要\n")
log(f"- **構面**：平行分析建議 {n_pa} 因素；4 因素 EFA 中「鼓勵上網」與陪伴題同一因素（負荷 {e2_load:.2f}；半樣本 {ea_load:.2f}；"
    f"國七同義題 {e1_load:.2f}），「監看」兩題分散（i 偏擔憂、j 偏衝突）。CFA：M5（鼓勵併入陪伴的五因素）CFI={m5r.CFI:.3f}、TLI={m5r.TLI:.3f}、"
    f"RMSEA={m5r.RMSEA:.3f}、SRMR={m5r.SRMR:.3f}（驗證半樣本 RMSEA={m5b.RMSEA:.3f}；國七 B 卷 RMSEA={m5w1.RMSEA:.3f}、CFI={m5w1.CFI:.3f}），"
    f"依慣例屬配適不佳，只是相對優於其他模型（BIC_rel {m5r.BIC_rel:.0f} vs EFA 四因素 {m4r.BIC_rel:.0f}）。構面視為實用分組；"
    f"監看 α={a_mon:.2f}、與衝突潛在相關 {r_mc:.2f}，解讀需保守。迴歸中「鼓勵上網」與陪伴分開（見下）。")
log(f"- **主模型（一次一種數位教養 + 一般親子）**：沒有任何數位教養與較少成癮或較少憂鬱相關。"
    f"規範限制 成癮 {cb('主模型', 'addict_2', 'restrict')}、憂鬱 {cb('主模型', 'cesd_nosleep_2', 'restrict')}；"
    f"衝突責罵 成癮 {cb('主模型', 'addict_2', 'conflict')}、憂鬱 {cb('主模型', 'cesd_nosleep_2', 'conflict')}；"
    f"擔憂叮嚀 成癮 {cb('主模型', 'addict_2', 'worry')}；監看 憂鬱 {cb('主模型', 'cesd_nosleep_2', 'monitor')}；"
    f"鼓勵上網 成癮 {cb('主模型', 'addict_2', 'encourage')}。")
log(f"- **規範限制（規定時段、限制時長）↔ 較早睡、睡較多**：主模型 上床 {cb('主模型', 'bedtime_2', 'restrict')}（約早 "
    f"{-b_rb * sd_bed * 60:.0f} 分鐘/SD）；睡眠 {cb('主模型', 'sleep_hrs_2', 'restrict')}；再控制國七作息（A 卷；教養仍為國八測量）"
    f"上床 {cb('主模型+控制W1結果', 'bedtime_2', 'restrict')}、睡眠 {cb('主模型+控制W1結果', 'sleep_hrs_2', 'restrict')}。"
    f"不論怎麼調整都穩定（僅共變項 {ad('bedtime_2', '僅共變項')} 至全部互相控制 {ad('bedtime_2', '全部互相控制')}；圖 S2）。")
log(f"- **但「規則→早睡」沒有前瞻證據**：國七 B 卷 規則→國八上床 {cb('W1教養_未控W1作息(B卷)', 'bedtime_2', 'restrict')}，"
    f"但 B 卷無國七作息題、未控基線。時序負對照（A 卷同一批 n={nc('bedtime_2', '→國八(同期)').n:,}）：國八規則與「一年前」國七上床 "
    f"{ncf('bedtime_2', '→國七(一年前)')}，和與同期國八上床 {ncf('bedtime_2', '→國八(同期)')} 一樣強；睡眠 {ncf('sleep_hrs_2', '→國七(一年前)')} vs "
    f"{ncf('sleep_hrs_2', '→國八(同期)')} → 關聯在一年前就已存在，反映家庭間穩定差異（可能是早已存在的規則，也可能是穩定混淆），"
    f"B 卷前瞻係數很可能含基線作息差異。A+B 卷小樣本（n={ab('bedtime_2').n}，能同時控基線）：主模型設定下 規則→國八上床 "
    f"控國七成癮憂鬱 vs 控國七上床 {abf(ab('bedtime_2'))}，兩者都≈0，樣本太小無法裁決；互相控制設定下 {abf(ab('bedtime_2', '互相控制(敏感度)'))}、"
    f"睡眠 {abf(ab('sleep_hrs_2', '互相控制(敏感度)'))}。")
log(f"- **規則與成癮/憂鬱：總關聯≈0；互相控制下的「好處」可能是碰撞點假象**：互相控制所有數位教養時才變負（成癮 "
    f"{cb('敏感度_六構面互相控制', 'addict_2', 'restrict', True)}、憂鬱 {cb('敏感度_六構面互相控制', 'cesd_nosleep_2', 'restrict', True)}），"
    f"主要來自控制衝突責罵（+衝突：成癮 {ad('addict_2', '+一般親子+衝突責罵')}、憂鬱 {ad('cesd_nosleep_2', '+一般親子+衝突責罵')}）"
    f"與擔憂叮嚀（成癮 {ad('addict_2', '+一般親子+擔憂叮嚀')}）。以相關分解：r(規則,衝突)={ca['r(規則,衝突)']:.2f}、r(衝突,成癮)={ca['r(衝突,Y)']:.2f}、"
    f"r(規則,成癮)={ca['r(規則,Y)']:+.3f} → β規則|衝突={ca['β規則|衝突']:+.3f}（邊際 {ca['來自規則邊際相關']:+.3f}、衝突路徑 {ca['來自衝突路徑']:+.3f}）；"
    f"憂鬱 {cc_['β規則|衝突']:+.3f}（衝突路徑 {cc_['來自衝突路徑']:+.3f}）。責罵題（因上網責罵、網購衝突）可能是孩子問題上網的結果 → "
    f"不宜寫成「有規則、不責罵就較不成癮/憂鬱」。規則×衝突交互作用（成癮 β_int={rca.beta_int:+.3f} [{rca.lo:+.3f}, {rca.hi:+.3f}]）"
    f"同樣以碰撞點為條件，只列敏感度。再控國七成癮後，國八規則與成癮反而正相關 {cb('主模型+控制W1結果', 'addict_2', 'restrict')}，"
    f"可能是同一年內的反應（成癮上升時父母加立規則）；但交叉延宕的國七成癮→國八規則為負，時間尺度不同，方向無法確定。")
log(f"- **鼓勵上網 vs 陪伴引導**：同一因素（r=.64），但效標關聯不同。主模型 成癮：鼓勵 {cb('主模型', 'addict_2', 'encourage')}、"
    f"陪伴(cdf) {cb('主模型', 'addict_2', 'active')}；合併(cdef) {cb('敏感度_陪伴含鼓勵(單一+一般親子)', 'addict_2', 'active5')}（鼓勵的訊號被稀釋）。"
    f"再控國七成癮：鼓勵 {cb('主模型+控制W1結果', 'addict_2', 'encourage')}。兩者互相控制時方向相反（陪伴 "
    f"{cb('敏感度_六構面互相控制', 'addict_2', 'active', True)}、鼓勵 {cb('敏感度_六構面互相控制', 'addict_2', 'encourage', True)}），"
    f"但那是彼此條件化的估計；國七 B 卷前瞻兩者皆不顯著（陪伴 {cb('W1前瞻(B卷,控W1結果)', 'addict_2', 'active', True)}、"
    f"鼓勵 {cb('W1前瞻(B卷,控W1結果)', 'addict_2', 'encourage', True)}）。")
log(f"- **陪伴引導的好處被一般親子關係解釋掉**：只控共變項時四項皆有利（上床 {cb('單一_僅共變項', 'bedtime_2', 'active', True)}、"
    f"成癮 {cb('單一_僅共變項', 'addict_2', 'active', True)}、憂鬱 {cb('單一_僅共變項', 'cesd_nosleep_2', 'active', True)}）；"
    f"控制溫暖/敵意/凝聚後全部消失，憂鬱甚至略正 {cb('主模型', 'cesd_nosleep_2', 'active')}（可能是對有狀況孩子的反應性投入，"
    f"或控制溫暖後殘差意義改變；不宜解讀為有害）。")
log(f"- **一般親子關係比任何數位教養都重要**：憂鬱 家庭凝聚 {cb('主模型', 'cesd_nosleep_2', 'cohesion')}、敵意/冷漠 "
    f"{cb('主模型', 'cesd_nosleep_2', 'hostile')}；成癮 敵意/冷漠 {cb('主模型', 'addict_2', 'hostile')}。"
    "六種數位教養一起加入、超越一般親子的 ΔR²："
    + "；".join(f"{OUTC[r_.outcome]} {r_['ΔR2_數位教養(超越一般親子)']:.3f}" for _, r_ in r2.iterrows()) + "。")
log(f"- **衝突責罵**：主模型 成癮 {cb('主模型', 'addict_2', 'conflict', True)}、憂鬱 {cb('主模型', 'cesd_nosleep_2', 'conflict', True)}"
    f"（上床 {cb('主模型', 'bedtime_2', 'conflict', True)} 不顯著）；再控國七同一結果 成癮 {cb('主模型+控制W1結果', 'addict_2', 'conflict', True)}、"
    f"憂鬱 {cb('主模型+控制W1結果', 'cesd_nosleep_2', 'conflict', True)}。前瞻：單一構面交叉延宕 親→子 憂鬱 {cl('conflict', '憂鬱', '親→子')}、"
    f"成癮 {cl('conflict', '網路成癮', '親→子')}；子→親 憂鬱 {cl('conflict', '憂鬱', '子→親')}、成癮 {cl('conflict', '網路成癮', '子→親')}；"
    f"兩方向差異不顯著（憂鬱 {cd('conflict', '憂鬱')}；成癮 {cd('conflict', '網路成癮')}）→ 無法判斷哪個方向為主。"
    f"再控國七一般親子後（2d）前瞻係數變小且 FDR 不顯著：成癮 {cb('W1前瞻(B卷,控W1結果)', 'addict_2', 'conflict')}、"
    f"憂鬱 {cb('W1前瞻(B卷,控W1結果)', 'cesd_nosleep_2', 'conflict')}。")
log(f"- **擔憂叮嚀**：主模型 成癮 {cb('主模型', 'addict_2', 'worry')}；交叉延宕 親→子 國七擔憂→國八成癮 {cl('worry', '網路成癮', '親→子')}"
    f"（兩方向差 {cd('worry', '網路成癮')}，FDR 後不顯著）；控國七一般親子後 {cb('W1前瞻(B卷,控W1結果)', 'addict_2', 'worry')}。")
log(f"- **緩衝（調節）**：FDR 校正後，沒有任何教養方式削弱「成癮→憂鬱」。同期顯著者皆為放大："
    + ("；".join(f"{XLABS[r_.X]}×{DPLAB[r_.M.replace('dp_', '')[:-2]]} β_int={r_.beta_int:+.3f} [{r_.lo:+.3f}, {r_.hi:+.3f}], "
                f"{pfmt(r_.p)}, q={r_.q:.4f}, n={r_.n:,}" for _, r_ in isame.iterrows()) or "（無）")
    + f"；前瞻版本 {len(pint)} 項全不顯著（|β_int|≤{pint.beta_int.abs().max():.3f}，最小 q={pint.q.min():.3f}）。")
log(f"- **反向因果（子→親）**：國七憂鬱→國八規範限制 {cl('restrict', '憂鬱', '子→親')}；國七成癮→國八規範限制 "
    f"{cl('restrict', '網路成癮', '子→親')}；國七憂鬱→國八陪伴 {cl('active', '憂鬱', '子→親')}。規則–憂鬱兩方向差異 "
    f"{cd('restrict', '憂鬱')}（子→親為主，FDR q={cdiff[(cdiff['教養'] == 'restrict') & (cdiff['結果'] == '憂鬱')].q.iloc[0]:.3f}）。"
    f"國七規則→國八憂鬱（控國七憂鬱與一般親子）{cb('W1前瞻(B卷,控W1結果)', 'cesd_nosleep_2', 'restrict')}。")

(OUT / "results.md").write_text("\n".join(LOG) + "\n", encoding="utf-8")
print("\n完成：", OUT)
