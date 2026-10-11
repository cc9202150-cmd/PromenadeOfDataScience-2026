"""國七晚間時間日誌的類型學（W1 A 卷）→ 一年後（國八）的身心與作息結果。

步驟：
  1. 18:00–02:00 八個逐時「主要活動」→ 7 種狀態序列；處理遺漏時段。
     另以「主要或次要活動」（s70 每時段可複選次要活動）建構逐時「任何螢幕使用」。
  2. 四種分群法（k = 3–8）：
       (a) 潛在類別分析 LCA（類別型混合模型，EM，200 組隨機起始值，BIC）← 主解；
       (b) PAM（k-medoids）於「動態 Hamming 距離」DHD（Lesnard 2010：時段別、依轉移率的替換成本）；
       (c) Ward 階層式分群（DHD）；
       (d) k-modes（一般 Hamming）。
     以 ASW、BIC、學校叢集 bootstrap（LCA 與 PAM 皆從頭重新擬合）、跨方法 ARI 與可解釋性選 k。
     最終採 k=6：所有輪廓係數與 bootstrap 指標皆優於 k=7；k=7 多出的「補習後晚睡型」在 bootstrap 中多半解散，
     改以規則式切分（補習類中 23:00 是否仍醒、深夜是否用螢幕）分析，k=7 列為敏感度。
  3. 加權群大小、逐時活動組成（小倍數圖）、索引圖、國七共變項輪廓。
  4. 類型 → 國八結果（加權 WLS，學校叢集標準誤，BH-FDR）：M1/M2/M3；
     敏感度（k=7、含自願加答者、排除不確定歸類）；「深夜本身 vs 螢幕」的直接檢驗
     （類型 vs 國七上床時間的巢套比較、上床時間與深夜螢幕同時進入、補習類依深夜作息拆分）。

樣本：分群使用全部有效日誌（含事後自願加答 A 卷者，增加資訊、以人為單位不加權）；
占比、輪廓與國八分析只用課堂作答 A 卷者（題本以學校為單位分派，89 校 A 卷、90 校 B 卷）（response 1 = 課堂 A 卷、3 = 僅 A 卷）；
response 2/5 是課堂作答 B 卷後自願於暑假前後加答 A 卷，非隨機，只列入敏感度分析。描述與迴歸用 HOUWGT。
執行：.venv/Scripts/python -I scripts/02_evening_typology.py   （約 15–20 分鐘）
輸出：outputs/typology/figures/*.png、outputs/typology/tables/*.csv、outputs/typology/results.md
"""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from kmodes.kmodes import KModes
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import silhouette_score
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tigps import data as D  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
OUT = ROOT / "outputs" / "typology"
FIG, TAB = OUT / "figures", OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
SEED = 20261019
KS = list(range(3, 9))
N_BOOT, N_START_BOOT = 60, 30   # bootstrap 次數；每次 LCA 從頭以 30 組隨機起始值擬合（不以原解暖啟動）
N_START = 200
K_FINAL, K_ALT, METHOD = 6, 7, "lca"   # K_ALT：曾考慮的 k=7（多分出補習後晚睡型），列為敏感度
INK, MUTED, GRID = "#1f1f1f", "#6b6a66", "#e6e4de"
LOG: list[str] = []
T0 = time.time()


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def wmean(x, w):
    m = x.notna() & w.notna()
    return np.average(x[m], weights=w[m]) if m.any() else np.nan


def wls_cluster(formula, d, w="HOUWGT", g="nschool_id"):
    return smf.wls(formula, d, weights=d[w]).fit(cov_type="cluster", cov_kwds={"groups": d[g]})


def wald_p(f, rows):
    """rows：[{參數名: 係數}, ...]，檢定各列線性組合同時為 0。"""
    names = list(f.params.index)
    R = np.zeros((len(rows), len(names)))
    for i, r in enumerate(rows):
        for p, c in r.items():
            R[i, names.index(p)] = c
    return float(f.wald_test(R, scalar=True).pvalue)


# ---------- 1. 序列建構 ----------
# 7 種狀態：小類（運動社團才藝、與家人朋友聊天、其他；各時段 ≤4%，多數 <3%）併入「生活與其他」
STATES = ["補習課輔", "寫作業讀書", "生活與其他", "螢幕：影音", "螢幕：遊戲電視", "螢幕：社群聊天", "睡覺"]
GROUP2STATE = {"補習/課輔": "補習課輔", "寫作業讀書": "寫作業讀書", "運動社團才藝": "生活與其他",
               "吃飯家事生活": "生活與其他", "與家人朋友聊天": "生活與其他", "其他": "生活與其他",
               "螢幕：影音": "螢幕：影音", "螢幕：遊戲/電視": "螢幕：遊戲電視",
               "螢幕：社群聊天": "螢幕：社群聊天", "睡覺": "睡覺"}
STATE_LABEL = {"補習課輔": "補習／課輔", "寫作業讀書": "寫作業／讀書", "生活與其他": "吃飯盥洗家事等生活",
               "螢幕：影音": "螢幕：影音音樂", "螢幕：遊戲電視": "螢幕：遊戲／電視",
               "螢幕：社群聊天": "螢幕：社群聊天", "睡覺": "睡覺"}
# 色盤：以 OKLab + Machado(2009) 色盲模擬檢查堆疊相鄰色差（正常視覺 ≥15、三型色盲 ≥9.6）
COLORS = ["#1baf7a", "#2a78d6", "#d9d5cb", "#f2b134", "#e3592f", "#a8326e", "#2e2a6b"]
SLOTS = list(D.DIARY_SLOTS)                       # 18 … 01
HOURS = ["18", "19", "20", "21", "22", "23", "0", "1"]
SEC_LETTER = dict(zip(SLOTS, "bdfhjlnp"))        # 次要活動 as70b1–b21、as70d1–d21 …（0/1 複選）
SCR_CODES = [c for c, g in D.DIARY_GROUPS.items() if g.startswith("螢幕")]   # 14–19
LATE_SLOTS = ["23", "00", "01"]                    # 23:00–02:00
SI = {s: i for i, s in enumerate(STATES)}
SLEEP = SI["睡覺"]
SCREEN = [SI["螢幕：影音"], SI["螢幕：遊戲電視"], SI["螢幕：社群聊天"]]

raw1, _ = D.read_wave(D.W1_SAV)
w1 = D.build_w1(raw1)
w1["response"] = raw1["response"]
A = w1[w1["has_A"]].copy()
A["randA"] = A["response"].isin([1, 3])           # 課堂作答 A 卷（以學校為單位分派）（2、5 為事後自願加答）
for h in SLOTS:  # 任何螢幕使用 = 主要活動為螢幕，或次要活動勾選任一螢幕項目
    sec = raw1.loc[A.index, [f"as70{SEC_LETTER[h]}{c}" for c in SCR_CODES]].eq(1).any(axis=1)
    A[f"scr_any_{h}"] = (A[f"diary_{h}"].isin(SCR_CODES) | sec).astype(float)
seq = pd.DataFrame({h: A[f"diary_{h}"].map(D.DIARY_GROUPS).map(GROUP2STATE).map(SI) for h in SLOTS},
                   index=A.index)
say("== 1. 序列建構（W1 A 卷，上學日晚間 18:00–02:00 逐時主要活動）==")
say(f"W1 A 卷 n={len(A)}（response：{A['response'].value_counts().sort_index().to_dict()}；"
    "1/3 = 課堂作答（以學校為單位分派），2/5 = 事後自願加答）；每人遺漏時段數分布：",
    seq.isna().sum(axis=1).value_counts().sort_index().to_dict())
grp_share = pd.DataFrame({h: A[f"diary_{h}"].map(D.DIARY_GROUPS) for h in SLOTS}).apply(
    lambda s: s.value_counts(normalize=True)).fillna(0)
SMALL = ["運動社團才藝", "與家人朋友聊天", "其他"]
small_max = (100 * grp_share.loc[SMALL].max(axis=1)).round(1).to_dict()
say("併入「生活與其他」的小類，各時段最大占比 %：", small_max)
all_na = seq.isna().all(axis=1)
# 規則 1：前一時段為「睡覺」而本時段空白 → 視為仍在睡（入睡後常不再填寫）
n_r1 = 0
for i in range(1, len(SLOTS)):
    fill = seq.iloc[:, i].isna() & (seq.iloc[:, i - 1] == SLEEP)
    n_r1 += int(fill.sum())
    seq.loc[fill, SLOTS[i]] = SLEEP
# 規則 2：其餘遺漏 ≤2 個時段者，以前一時段（無前一時段者以後一時段）補值；>2 者排除
n_left = seq.isna().sum(axis=1)
ok = ~all_na & (n_left <= 2)
n_r2 = int(n_left[ok].sum())
n_drop = int((~all_na & (n_left > 2)).sum())
seq = seq[ok].ffill(axis=1).bfill(axis=1).astype(int)
A = A.loc[seq.index]
raw_diary = A[[f"diary_{h}" for h in SLOTS]]
A["straight"] = (raw_diary.nunique(axis=1) == 1) & (raw_diary.iloc[:, 0] != 8)  # 8 格同一非睡眠代碼
say(f"全空白排除 {int(all_na.sum())}；規則1（睡後空白→睡覺）補 {n_r1} 格；規則2（相鄰時段補值）補 {n_r2} 格；"
    f"補值後仍遺漏 >2 格排除 {n_drop}；分群樣本 n={len(seq)}（其中課堂 A 卷 {int(A['randA'].sum())}）")
say(f"8 格皆為同一非睡眠代碼（疑似直線作答）{int(A['straight'].sum())} 人（保留，敏感度分析排除）")
X = seq.to_numpy()
N, T = X.shape
S = len(STATES)
A["late_screen_main"] = np.isin(X[:, 5:], SCREEN).any(1).astype(float)
A["late_screen_any"] = A[[f"scr_any_{h}" for h in LATE_SLOTS]].max(axis=1)
A["awake23"] = (X[:, 5] != SLEEP).astype(float)
U, inv, cnt = np.unique(X, axis=0, return_inverse=True, return_counts=True)
inv, cnt = inv.ravel(), cnt.astype(float)
say(f"唯一序列 {len(U)} 種（LCA、PAM 以唯一序列 × 次數計算，與逐人計算等價）")


# ---------- 2. 距離與分群 ----------
def dhd_costs(X):
    """動態 Hamming（Lesnard 2010）：c_t(a,b) = 4 − p_{t-1}(a,b) − p_{t-1}(b,a) − p_t(a,b) − p_t(b,a)，
    p_t(a,b) = P(t+1 時段為 b | t 時段為 a)；首末時段只有一側轉移，成本 ×2。"""
    P = []
    for t in range(T - 1):
        m = np.zeros((S, S))
        np.add.at(m, (X[:, t], X[:, t + 1]), 1)
        p = m / np.maximum(m.sum(1, keepdims=True), 1)
        P.append(p + p.T)
    C = [2 * (2 - P[0])] + [4 - P[t - 1] - P[t] for t in range(1, T - 1)] + [2 * (2 - P[-1])]
    for c in C:
        np.fill_diagonal(c, 0)
    return np.array(C)


def seq_dist(U, C):
    return sum(C[t][U[:, t][:, None], U[:, t][None, :]] for t in range(T))


C_DHD = dhd_costs(X)
Du = seq_dist(U, C_DHD)
Hu = seq_dist(U, np.array([1 - np.eye(S)] * T))
Dfull, Hfull = Du[inv][:, inv], Hu[inv][:, inv]
pd.DataFrame(C_DHD.mean(0), index=STATES, columns=STATES).round(2).to_csv(
    TAB / "dhd_mean_substitution_cost.csv", encoding="utf-8-sig")


def pam(Dm, w, k, max_iter=300):
    """加權 PAM（BUILD + SWAP）；Dm 為唯一序列距離、w 為次數權重。回傳 medoid 索引。"""
    med = [int(np.argmin(Dm @ w))]
    d1 = Dm[med[0]].copy()
    for _ in range(1, k):
        gain = np.maximum(d1[None, :] - Dm, 0) @ w
        gain[med] = -1
        med.append(int(np.argmax(gain)))
        d1 = np.minimum(d1, Dm[med[-1]])
    med, ar = np.array(med), np.arange(len(w))
    for _ in range(max_iter):
        Dk = Dm[med]
        order = np.argsort(Dk, axis=0, kind="stable")
        near, d1, d2 = order[0], Dk[order[0], ar], Dk[order[1], ar]
        dwo = np.where(near[None, :] == np.arange(k)[:, None], d2[None, :], d1[None, :])
        best, bh, bi = -1e-9, None, None
        for s in range(0, len(w), 256):
            delta = (np.minimum(Dm[s:s + 256][:, None, :], dwo[None]) - d1[None, None, :]) @ w
            j = np.unravel_index(np.argmin(delta), delta.shape)
            if delta[j] < best:
                best, bh, bi = delta[j], s + j[0], j[1]
        if bh is None:
            break
        med[bi] = bh
    return med


Yu = np.eye(S)[U].reshape(len(U), T * S)  # 唯一序列 one-hot 攤平：n×(T·S)


def lca_post(pi, th):
    logp = np.log(pi)[None, :] + Yu @ np.log(th).reshape(len(pi), T * S).T
    mx = logp.max(1, keepdims=True)
    lse = mx[:, 0] + np.log(np.exp(logp - mx).sum(1))
    return np.exp(logp - lse[:, None]), lse


def lca(w, K, n_start, rng, max_iter=3000, tol=1e-8, diag=False):
    """類別型潛在類別分析（EM，隨機起始值）。回傳 dict(ll, r 後驗, pi, th)；
    diag=True 另回傳：與最佳解差距 <1 個對數概似單位的起始解數，及其分類與最佳解的最小 ARI（解是否被唯一辨識）。"""
    starts = [(np.full(K, 1 / K), rng.dirichlet(np.ones(S), size=(K, T))) for _ in range(n_start)]
    fits = []
    for pi, th in starts:
        old = -np.inf
        for _ in range(max_iter):
            r, lse = lca_post(pi, th)
            ll = w @ lse
            wr = r * w[:, None]
            pi = wr.sum(0) / wr.sum()
            th = (wr.T @ Yu).reshape(K, T, S) + 1e-6
            th /= th.sum(2, keepdims=True)
            if ll - old < tol * abs(ll):
                break
            old = ll
        fits.append((ll, pi, th))
    lls = np.array([f[0] for f in fits])
    ll, pi, th = fits[int(np.argmax(lls))]
    out = dict(ll=ll, r=lca_post(pi, th)[0], pi=pi, th=th)
    if diag:
        lab0 = out["r"].argmax(1)[inv]
        near = [f for f in fits if f[0] > ll - 1]
        out["n_near"] = len(near)
        out["near_ari_min"] = min(ARI(lab0, lca_post(f[1], f[2])[0].argmax(1)[inv]) for f in near)
    return out


def relabel_by_size(lab):
    sz = pd.Series(lab).value_counts()
    return pd.Series(lab).map({c: i for i, c in enumerate(sz.index)}).to_numpy()


say("\n== 2. 分群比較（k=3–8）==")
rng = np.random.default_rng(SEED)
Z_ward = linkage(squareform(Dfull, checks=False), method="ward")
sols, rows = {}, []
for k in KS:
    med = pam(Du, cnt, k)
    lab_pam = relabel_by_size(np.argmin(Du[med], axis=0)[inv])
    lab_ward = relabel_by_size(fcluster(Z_ward, k, criterion="maxclust") - 1)
    km = KModes(n_clusters=k, init="Huang", n_init=10, random_state=SEED).fit(X)
    lab_km = relabel_by_size(km.labels_)
    fit = lca(cnt, k, N_START, rng, diag=True)
    lab_lca = relabel_by_size(fit["r"].argmax(1)[inv])
    r_ind = fit["r"][inv]
    npar = (k - 1) + k * T * (S - 1)
    sols[k] = dict(pam=lab_pam, ward=lab_ward, kmodes=lab_km, lca=lab_lca, lca_fit=fit, medoids=med)
    rows.append(dict(
        k=k, ASW_PAM=silhouette_score(Dfull, lab_pam, metric="precomputed"),
        ASW_Ward=silhouette_score(Dfull, lab_ward, metric="precomputed"),
        ASW_LCA=silhouette_score(Dfull, lab_lca, metric="precomputed"),
        ASW_kmodes=silhouette_score(Dfull, lab_km, metric="precomputed"),
        ASW_kmodes_Hamming=silhouette_score(Hfull, lab_km, metric="precomputed"),
        kmodes_cost=km.cost_, LCA_LL=fit["ll"], LCA_BIC=-2 * fit["ll"] + npar * np.log(N),
        LCA_entropyR2=1 + (r_ind * np.log(np.clip(r_ind, 1e-300, 1))).sum() / (N * np.log(k)),
        LCA_starts_within1LL=f"{fit['n_near']}/{N_START}", LCA_near_best_minARI=fit["near_ari_min"],
        LCA_min_class_pct=100 * np.bincount(lab_lca).min() / N,
        ARI_LCA_PAM=ARI(lab_lca, lab_pam), ARI_LCA_kmodes=ARI(lab_lca, lab_km),
        ARI_LCA_Ward=ARI(lab_lca, lab_ward), ARI_PAM_Ward=ARI(lab_pam, lab_ward)))
say(f"（各 k 擬合完成，{time.time() - T0:.0f}s）")
ksel = pd.DataFrame(rows).set_index("k")
ABBR = ["補", "讀", "生", "影", "遊", "社", "睡"]
for k in KS:  # 供判讀：各 k 下 LCA 各類每時段最常見狀態（占比）與午夜仍醒比例
    say(f"-- LCA k={k} --")
    for c in range(k):
        Xi = X[sols[k]["lca"] == c]
        modal = [np.bincount(Xi[:, j], minlength=S) for j in range(T)]
        say(f"  類{c} n={len(Xi):4d} 午夜醒 {np.mean(Xi[:, 6] != SLEEP):.0%} | " +
            " ".join(f"{ABBR[m.argmax()]}{m.max() / len(Xi):.0%}" for m in modal))

# 學校叢集 bootstrap：重抽學校 → 唯一序列權重 → 從頭重新擬合（LCA 隨機起始、PAM 確定性 BUILD）
# → 原樣本依新解指派 → 與原解比 ARI、各群 Jaccard
school = A["nschool_id"].to_numpy()
schools = np.unique(school)
idx_by_school = {s: np.flatnonzero(school == s) for s in schools}
boot = {(m, k): dict(ari=[], jac=[]) for m in ("lca", "pam") for k in KS}


def jaccard_max(lab0, labb, k):
    return [max(((lab0 == c) & (labb == c2)).sum() / ((lab0 == c) | (labb == c2)).sum()
                for c2 in np.unique(labb)) for c in range(k)]


for b in range(N_BOOT):
    mult = np.zeros(N)
    for s in rng.choice(schools, size=len(schools), replace=True):
        mult[idx_by_school[s]] += 1
    wb = np.bincount(inv, weights=mult, minlength=len(U))
    for k in KS:
        fb = lca(wb, k, N_START_BOOT, rng, tol=1e-7)
        lab_b = fb["r"].argmax(1)[inv]
        boot["lca", k]["ari"].append(ARI(sols[k]["lca"], lab_b))
        boot["lca", k]["jac"].append(jaccard_max(sols[k]["lca"], lab_b, k))
        lab_b = np.argmin(Du[pam(Du, wb, k)], axis=0)[inv]
        boot["pam", k]["ari"].append(ARI(sols[k]["pam"], lab_b))
        boot["pam", k]["jac"].append(jaccard_max(sols[k]["pam"], lab_b, k))
    if (b + 1) % 10 == 0:
        print(f"  bootstrap {b + 1}/{N_BOOT}（{time.time() - T0:.0f}s）", flush=True)
for m in ("lca", "pam"):
    ksel[f"boot_ARI_{m}"] = [np.mean(boot[m, k]["ari"]) for k in KS]
    ksel[f"boot_ARI_{m}_p05"] = [np.percentile(boot[m, k]["ari"], 5) for k in KS]
    ksel[f"boot_minJaccard_{m}"] = [np.mean(boot[m, k]["jac"], axis=0).min() for k in KS]
say(ksel.drop(columns=["LCA_LL"]).round(3).T.to_string())
ksel.round(4).to_csv(TAB / "k_selection.csv", encoding="utf-8-sig")
ASW_COLS = ["ASW_LCA", "ASW_PAM", "ASW_Ward", "ASW_kmodes"]
best_k = {c: int(ksel[c].idxmax()) for c in ASW_COLS + ["boot_ARI_lca", "boot_minJaccard_lca"]}
best_k["LCA_BIC"] = int(ksel["LCA_BIC"].idxmin())
say("各指標最佳 k：", best_k)
CMP_IDX = ASW_COLS + ["boot_ARI_lca", "boot_minJaccard_lca"]
fin_better = [c for c in CMP_IDX if ksel.loc[K_FINAL, c] > ksel.loc[K_ALT, c]]
say(f"k={K_FINAL} vs k={K_ALT}：{len(fin_better)}/{len(CMP_IDX)} 項指標（ASW 四法、LCA bootstrap ARI 與最低 Jaccard）"
    f"以 k={K_FINAL} 較佳：{fin_better}；BIC 差 {ksel.loc[K_FINAL, 'LCA_BIC'] - ksel.loc[K_ALT, 'LCA_BIC']:+.0f}")

# k-modes 種子穩定度（最終 k）
km_runs = [KModes(n_clusters=K_FINAL, init="Huang", n_init=1, random_state=SEED + i).fit(X).labels_
           for i in range(10)]
km_seed_ari = np.mean([ARI(a, b) for i, a in enumerate(km_runs) for b in km_runs[i + 1:]])
say(f"k-modes（k={K_FINAL}）10 組隨機種子兩兩 ARI 平均 = {km_seed_ari:.3f}")


# ---------- 2b. 最終解與命名 ----------
def share(lab, c, states, slots):
    return np.isin(X[lab == c][:, slots], [SI[s] for s in states]).mean()


RULES = [  # (名稱, 條件, 排序依據)：依序貪婪指派，每類只用一次
    ("作息紊亂型", lambda f: f.life_late > .4, "life_late"),
    ("補習後晚睡型", lambda f: (f.cram > .3) & (f.awake23 > .8), "awake23"),
    ("補習晚歸型", lambda f: f.cram > .5, "cram"),
    ("電玩遊戲型", lambda f: f.game > .4, "game"),
    ("影音社群夜貓型", lambda f: f.vidsoc > .3, "vidsoc"),
    ("回家讀書型", lambda f: f.study > .4, "study"),
    ("早睡規律型", lambda f: f.sleep22 > .6, "sleep22"),
]
ORDER0 = ["早睡規律型", "回家讀書型", "補習晚歸型", "補習後晚睡型", "電玩遊戲型", "影音社群夜貓型", "作息紊亂型"]
LATE_ALL = ["補習後晚睡型", "電玩遊戲型", "影音社群夜貓型"]   # 晚睡型（作息紊亂型另論）
CL, CRAM, CHAOS = "補習後晚睡型", "補習晚歸型", "作息紊亂型"


def name_types(lab, k):
    """依各類逐時組成（RULES）命名；回傳 {類: 名稱}、特徵表、排序後名稱清單。"""
    feat = pd.DataFrame({c: dict(
        cram=share(lab, c, ["補習課輔"], [0, 1, 2]), study=share(lab, c, ["寫作業讀書"], [1, 2, 3]),
        late_study=share(lab, c, ["寫作業讀書"], [3, 4, 5]),
        game=share(lab, c, ["螢幕：遊戲電視"], [0, 1, 2, 3, 4]),
        vidsoc=share(lab, c, ["螢幕：影音", "螢幕：社群聊天"], [0, 1, 2, 3, 4]),
        life_late=share(lab, c, ["生活與其他"], [5, 6, 7]), sleep22=share(lab, c, ["睡覺"], [4]),
        awake23=1 - share(lab, c, ["睡覺"], [5]), awake0=1 - share(lab, c, ["睡覺"], [6]))
        for c in range(k)}).T
    names = {}
    for nm, cond, key in RULES:
        cand = feat[cond(feat) & ~feat.index.isin(list(names))]
        if len(cand):
            names[int(cand[key].idxmax())] = nm
    for c in range(k):
        names.setdefault(c, f"類型{c}")
    order = [n for n in ORDER0 if n in names.values()]
    return names, feat, order + [n for n in names.values() if n not in order]


lab = sols[K_FINAL][METHOD]
names, feat, ORDER = name_types(lab, K_FINAL)
LATE = [t for t in LATE_ALL if t in ORDER]
raw_lab = sols[K_FINAL]["lca_fit"]["r"].argmax(1)[inv]
map_raw = pd.crosstab(lab, raw_lab).idxmax(axis=1).to_dict()   # 依大小重編號後的類 → 原 LCA 類
post = sols[K_FINAL]["lca_fit"]["r"][inv][:, [map_raw[c] for c in range(K_FINAL)]]
A["maxpost"] = post.max(1)
say(f"\n== 最終解：{METHOD.upper()} k={K_FINAL}（各類特徵：時段占比）==\n" +
    feat.assign(name=pd.Series(names)).round(2).to_string())
REF = ORDER[0]
A["type"] = pd.Categorical(pd.Series(lab, index=A.index).map(names), categories=ORDER)
pd.DataFrame(post, columns=[names[c] for c in range(K_FINAL)], index=A["nstudent_id"]).round(4).to_csv(
    TAB / "posterior_probs.csv", encoding="utf-8-sig")
say("各類平均最大後驗機率：", A.groupby("type", observed=True)["maxpost"].mean().round(3).to_dict())
jac_final = pd.Series(np.mean(boot[METHOD, K_FINAL]["jac"], axis=0)).rename(names)
say("最終解各類 bootstrap Jaccard（>0.75 穩定、0.6–0.75 可辨識、<0.5 解散）：", jac_final.round(3).to_dict())

# k=7（多分出補習後晚睡型）：同一規則命名，與主解對照；其多出的一類在 bootstrap 中的穩定度
names_alt, feat_alt, ORDER_ALT = name_types(sols[K_ALT][METHOD], K_ALT)
say(f"\n-- k={K_ALT} 各類特徵 --\n" + feat_alt.assign(name=pd.Series(names_alt)).round(2).to_string())
A["type_alt"] = pd.Categorical(pd.Series(sols[K_ALT][METHOD], index=A.index).map(names_alt), categories=ORDER_ALT)
x_alt = pd.crosstab(A["type_alt"], A["type"])
x_alt.to_csv(TAB / f"crosstab_k{K_ALT}_vs_k{K_FINAL}.csv", encoding="utf-8-sig")
say(f"LCA k={K_ALT}（列）× k={K_FINAL}（欄）交叉表（人數）：\n" + x_alt.to_string())
jac_alt_reps = pd.DataFrame(boot[METHOD, K_ALT]["jac"]).rename(columns=names_alt)
jac_alt = jac_alt_reps.mean()
say(f"k={K_ALT} 各類 bootstrap Jaccard：", jac_alt.round(3).to_dict())
CL_STAB = (dict(mean=jac_alt_reps[CL].mean(), lt50=(jac_alt_reps[CL] < .5).mean(), ge75=(jac_alt_reps[CL] >= .75).mean(),
                to_main=x_alt.loc[CL].idxmax(), to_main_pct=x_alt.loc[CL].max() / x_alt.loc[CL].sum())
           if CL in jac_alt_reps else None)
if CL_STAB:
    say(f"k={K_ALT} 的「{CL}」：bootstrap Jaccard 平均 {CL_STAB['mean']:.2f}；{CL_STAB['lt50']:.0%} 的重抽樣本 <0.5、"
        f"{CL_STAB['ge75']:.0%} ≥0.75；在 k={K_FINAL} 中 {CL_STAB['to_main_pct']:.0%} 歸入「{CL_STAB['to_main']}」")
A[["nstudent_id", "response", "randA", "type", "type_alt", "maxpost", "straight"]].to_csv(
    TAB / "type_assignment.csv", index=False, encoding="utf-8-sig")

# 跨方法對應（最終 k）
agree = pd.DataFrame({m1: {m2: ARI(sols[K_FINAL][m1], sols[K_FINAL][m2]) for m2 in ("lca", "pam", "ward", "kmodes")}
                      for m1 in ("lca", "pam", "ward", "kmodes")}).round(3)
say("\n跨方法 ARI（k=%d）：\n%s" % (K_FINAL, agree.to_string()))
agree.to_csv(TAB / "method_agreement_ARI.csv", encoding="utf-8-sig")
xt = pd.crosstab(A["type"], pd.Series(sols[K_FINAL]["pam"], index=A.index, name="PAM群"), normalize="index")
xt.round(3).to_csv(TAB / "crosstab_LCA_vs_PAM.csv", encoding="utf-8-sig")
rep = pd.DataFrame({"pam": xt.idxmax(axis=1), "share": xt.max(axis=1)})
rep["shared"] = [[t2 for t2 in ORDER if t2 != t and rep.loc[t2, "pam"] == rep.loc[t, "pam"]] for t in rep.index]
rep["own"] = (rep["share"] >= .7) & (rep["shared"].str.len() == 0)
say("各 LCA 類型在 PAM 的主要去處：\n" + rep.to_string())

# ---------- 3. 群大小、時間軸圖、輪廓（只用課堂 A 卷）----------
A_all, X_all = A, X
RA = A_all["randA"].to_numpy()
A, Xr = A_all[RA].copy(), X_all[RA]
w = A["HOUWGT"]
say(f"\n== 3. 以下描述與推論只用課堂 A 卷 n={len(A)}（排除事後自願加答 {int((~RA).sum())} 人）==")
vol = pd.crosstab(A_all["type"], A_all["randA"], normalize="columns").round(3)
say("類型分布（未加權）：課堂 A 卷（True）vs 自願加答（False）\n" + vol.to_string())
size_rows = []
for t in ORDER:
    d = A.assign(y=(A["type"] == t).astype(float)).dropna(subset=["HOUWGT"])
    f = wls_cluster("y ~ 1", d)
    lo, hi = f.conf_int().iloc[0]
    m = (A["type"] == t).to_numpy()
    g, gw = A.loc[m], w[m]
    size_rows.append(dict(type=t, n=int(m.sum()), pct_w=100 * f.params.iloc[0], ci_lo=100 * lo, ci_hi=100 * hi,
                          boot_jaccard=jac_final[t], mean_maxpost=g["maxpost"].mean(),
                          straight_pct=100 * g["straight"].mean(),
                          bed_before_midnight_pct=100 * (g["bedtime"] < 24)[g["bedtime"].notna()].mean(),
                          bed_missing_pct=100 * g["bedtime"].isna().mean(),
                          self_bed_midnight_pct_w=100 * wmean((g["bedtime"] >= 24).astype(float).where(
                              g["bedtime"].notna()), gw),
                          late_screen_main_pct=100 * wmean(g["late_screen_main"], gw),
                          late_screen_any_pct=100 * wmean(g["late_screen_any"], gw)))
sizes = pd.DataFrame(size_rows).set_index("type")

# 逐時組成（加權）＋任何螢幕使用（主要或次要）
comp, scr_h = {}, {}
for t in ORDER:
    m = (A["type"] == t).to_numpy() & w.notna().to_numpy()
    comp[t] = np.array([[np.average(Xr[m, j] == s, weights=w[m]) for j in range(T)] for s in range(S)])
    scr_h[t] = np.array([np.average(A.loc[m, f"scr_any_{h}"], weights=w[m]) for h in SLOTS])
pd.concat({t: pd.DataFrame(np.vstack([comp[t], scr_h[t]]), index=STATES + ["任何螢幕（主要或次要）"],
                           columns=[f"{h}:00" for h in HOURS]) for t in ORDER}).round(4).to_csv(
    TAB / "type_hourly_composition.csv", encoding="utf-8-sig")
sizes["awake_midnight_pct"] = [100 * (1 - comp[t][SLEEP, 6]) for t in ORDER]
sizes["awake_1am_pct"] = [100 * (1 - comp[t][SLEEP, 7]) for t in ORDER]
sizes["sleep_at_23_pct"] = [100 * comp[t][SLEEP, 5] for t in ORDER]
say("\n== 類型大小（加權 %，95% CI 為學校叢集）==\n" + sizes.round(2).to_string())
aw0 = pd.Series((Xr[:, 6] != SLEEP).astype(float), index=A.index)
ex = A["type"] != CHAOS
OVR = dict(awake0=100 * wmean(aw0, w), awake0_ex=100 * wmean(aw0[ex], w[ex]),
           self_bed=100 * wmean((A["bedtime"] >= 24).astype(float).where(A["bedtime"].notna()), w),
           late_main=100 * wmean(A["late_screen_main"], w), late_any=100 * wmean(A["late_screen_any"], w),
           scr_any_h={h: 100 * wmean(A[f"scr_any_{h}"], w) for h in SLOTS})
if CRAM in ORDER:  # 補習類中 23:00 仍醒者（規則式切分，取代不穩定的「補習後晚睡型」）
    mc = A["type"] == CRAM
    ma = mc & (A["awake23"] == 1)
    OVR.update(cram_awake=100 * wmean(A.loc[mc, "awake23"], w[mc]),
               cram_awake_scr=100 * wmean(A.loc[ma, "late_screen_any"], w[ma]),
               cram_awake_n=int(ma.sum()), cram_awake_midnight=100 * wmean(aw0[ma], w[ma]),
               cram_awake_bed=wmean(A.loc[ma, "bedtime"], w[ma]))
say(f"全體（加權）：日誌午夜 0:00–1:00 仍醒 {OVR['awake0']:.1f}%（排除{CHAOS} {OVR['awake0_ex']:.1f}%）；"
    f"自填上床時間 ≥ 午夜 {OVR['self_bed']:.1f}%；23:00–02:00 任一時段螢幕：只算主要活動 {OVR['late_main']:.1f}%、"
    f"主要或次要 {OVR['late_any']:.1f}%")
say("各時段任何螢幕使用（主要或次要）%：", {h: round(v, 1) for h, v in OVR["scr_any_h"].items()})
if "cram_awake" in OVR:
    say(f"{CRAM}中 23:00 仍醒 {OVR['cram_awake']:.1f}%（n={OVR['cram_awake_n']}）；其中深夜有螢幕 "
        f"{OVR['cram_awake_scr']:.1f}%、午夜仍醒 {OVR['cram_awake_midnight']:.1f}%、自填上床 {OVR['cram_awake_bed']:.2f} 時")
sizes.round(3).to_csv(TAB / "type_sizes.csv", encoding="utf-8-sig")
NUM_ZH = {6: "六", 7: "七", 5: "五", 8: "八"}


def fig_timeline():
    nc = 3 if len(ORDER) <= 6 else 4
    fig = plt.figure(figsize=(13, 7.4))
    gs = fig.add_gridspec(2, nc + 1 if nc == 3 else nc, width_ratios=[1] * nc + [0.82] if nc == 3 else None)
    axes = [fig.add_subplot(gs[i // nc, i % nc]) for i in range(len(ORDER))]
    lax = fig.add_subplot(gs[:, nc]) if nc == 3 else fig.add_subplot(gs[1, 3])
    x = np.arange(T)
    for i, (ax, t) in enumerate(zip(axes, ORDER)):
        bottom = np.zeros(T)
        for s in range(S):
            ax.bar(x, comp[t][s], bottom=bottom, width=0.92, color=COLORS[s], edgecolor="white", linewidth=0.7)
            bottom += comp[t][s]
        ax.plot(x, scr_h[t], color=INK, lw=1.4, marker="o", ms=4, mfc="white", mec=INK, mew=1.2, zorder=4)
        ax.set_xticks(x, HOURS, fontsize=9)
        ax.set_xlim(-0.55, T - 0.45)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, .5, 1], ["0", "50%", "100%"] if i % nc == 0 else [], fontsize=9)
        r = sizes.loc[t]
        ax.set_title(f"{t}", fontsize=13, fontweight="bold", color=INK, loc="left", pad=20)
        ax.text(0, 1.035, f"占 {r.pct_w:.1f}%（n={int(r.n):,}）｜午夜仍醒 {r.awake_midnight_pct:.0f}%",
                transform=ax.transAxes, fontsize=9.2, color=MUTED, va="bottom")
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)
        if i >= nc or i + nc >= len(ORDER):
            ax.set_xlabel("時段起點（時）", fontsize=9, color=MUTED)
    lax.axis("off")
    handles = [Patch(facecolor=COLORS[s], edgecolor="none", label=STATE_LABEL[STATES[s]]) for s in reversed(range(S))]
    handles.append(Line2D([], [], color=INK, lw=1.4, marker="o", ms=5, mfc="white", mec=INK,
                          label="任何螢幕使用\n（主要或次要活動）"))
    lax.legend(handles=handles, loc="center left" if nc == 3 else "upper left", frameon=False, fontsize=10.2,
               title="該時段主要活動", title_fontproperties={"weight": "bold", "size": 11}, handlelength=1.6,
               handleheight=1.1, borderaxespad=0.2, labelspacing=0.6, alignment="left")
    fig.text(0.01, 0.006, f"資料：TIGPS 2023 國七學生問卷，課堂作答 A 卷者（以學校為單位分派）（n={len(A):,}）；潛在類別分析（k={K_FINAL}）。"
             "長條＝各時段「主要活動」組成；折線＝該時段主要或次要活動含任何螢幕的比例；皆經 HOUWGT 加權。\n"
             "橫軸為每一小時時段的起點（0 = 午夜 0:00–1:00）。「遊戲／電視」為同一題目選項，無法區分。",
             fontsize=8.6, color=MUTED, va="bottom", linespacing=1.5)
    fig.suptitle(f"國七學生的{NUM_ZH.get(K_FINAL, K_FINAL)}種晚上：上學日 18:00–02:00 各時段主要活動組成",
                 fontsize=16, fontweight="bold", x=0.01, ha="left", y=0.99, color=INK)
    fig.tight_layout(rect=(0, 0.045, 1, 0.97), h_pad=2.0, w_pad=1.2)
    fig.savefig(FIG / "fig1_evening_types_timeline.png", dpi=200)
    plt.close(fig)


def fig_index():
    """序列索引圖：每列一位學生、每欄一個時段（全部分群樣本）。"""
    order_idx, bounds, cur = [], [], 0
    for t in ORDER:
        idx = np.flatnonzero((A_all["type"] == t).to_numpy())
        Xi = X_all[idx]
        first_sleep = np.where((Xi == SLEEP).any(1), (Xi == SLEEP).argmax(1), T)
        key = np.lexsort(tuple(Xi[:, j] for j in reversed(range(T))) + (first_sleep,))
        order_idx.extend(idx[key])
        bounds.append((cur, cur + len(idx), t))
        cur += len(idx)
    img = X_all[np.array(order_idx)]
    fig, ax = plt.subplots(figsize=(7.2, 8.6))
    ax.imshow(img, aspect="auto", cmap=ListedColormap(COLORS), vmin=-0.5, vmax=S - 0.5,
              interpolation="nearest", extent=(-0.5, T - 0.5, len(img), 0))
    for lo, hi, t in bounds:
        ax.axhline(hi, color="white", lw=2.2)
        ax.text(-0.75, (lo + hi) / 2, f"{t}\nn={hi - lo:,}", ha="right", va="center", fontsize=9.5, color=INK)
    ax.set_xticks(range(T), [f"{h}:00" for h in HOURS], fontsize=9)
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    handles = [Patch(facecolor=COLORS[s], label=STATE_LABEL[STATES[s]]) for s in range(S)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=4, frameon=False, fontsize=9)
    ax.set_title("每一列是一位國七學生的晚上（依類型分組、組內依入睡時段排序）", fontsize=12, loc="left",
                 color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig_s2_sequence_index.png", dpi=200)
    plt.close(fig)


fig_timeline()
fig_index()

# 國七共變項輪廓（加權平均；Wald 檢定各類平均相等，學校叢集）
A["digi_mature"] = (A["digi_level"] == "數位發展成熟區").astype(float)
A["private"] = (A["school_type"] == 2).astype(float)
PROF = {  # 變項: (標籤, 倍數, 方向：+1 越大越不利、−1 越大越有利、0 不涉好壞)
    "female": ("女生 %", 100, 0), "cram": ("有補習 %", 100, 0), "digi_mature": ("數位成熟區 %", 100, 0),
    "private": ("私立學校 %", 100, 0), "fam_econ": ("家庭經濟（1–4，大=富裕）", 1, -1),
    "grade_rank": ("成績排名（1–5，大=差）", 1, 1), "net_leisure": ("休閒上網指數（時，四用途加總†）", 1, 1),
    "addict": ("網路成癮（1–4）", 1, 1), "fomo": ("FOMO（1–4）", 1, 1), "bedtime": ("自填上床時間（時）", 1, 1),
    "sleep_hrs": ("自填睡眠時數", 1, -1), "cesd": ("憂鬱 CESD（0–48）", 1, 1)}
prof_rows, ptests, sds = {}, {}, {}
for v, (lbl, mult, _) in PROF.items():
    prof_rows[lbl] = {t: mult * wmean(A.loc[A["type"] == t, v], w[A["type"] == t]) for t in ORDER}
    prof_rows[lbl]["全體"] = mult * wmean(A[v], w)
    d = A[[v, "type", "HOUWGT", "nschool_id"]].dropna().copy()
    sds[lbl] = mult * np.sqrt(np.cov(d[v], aweights=d["HOUWGT"]))
    d["tc"] = d["type"].cat.codes
    f = wls_cluster(f"{v} ~ C(tc)", d)
    ptests[lbl] = wald_p(f, [{p: 1} for p in f.params.index if p.startswith("C(tc)")])
profile = pd.DataFrame(prof_rows).T[ORDER + ["全體"]]
profile["Wald_p"] = pd.Series(ptests)
say("\n== 國七輪廓（加權平均；Wald_p 為各類相等之檢定）==\n" + profile.round(3).to_string())
profile.round(4).to_csv(TAB / "type_profiles_w1.csv", encoding="utf-8-sig")


def fig_profile():
    z = pd.DataFrame({lbl: {t: (prof_rows[lbl][t] - prof_rows[lbl]["全體"]) / sds[lbl] for t in ORDER}
                      for lbl, _, _ in PROF.values()}).T
    dirs = [d for _, _, d in PROF.values()]
    div = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f3f2ee", "#e3592f"])
    neu = LinearSegmentedColormap.from_list("neu", ["#f3f2ee", "#a5a196"])
    rgba = np.array([div((dr * z.iloc[i].clip(-1, 1).to_numpy() + 1) / 2) if dr else
                     neu(z.iloc[i].abs().clip(0, 1).to_numpy()) for i, dr in enumerate(dirs)])
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.imshow(rgba, aspect="auto")
    for i, lbl in enumerate(z.index):
        for j, t in enumerate(ORDER):
            v = prof_rows[lbl][t]
            txt = f"{v:.0f}" if "%" in lbl else (f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=9.5,
                    color="white" if dirs[i] and abs(z.iat[i, j]) > 0.75 else INK)
    ax.set_xticks(range(len(ORDER)), ORDER, fontsize=10)
    ax.xaxis.tick_top()
    ylab = [f"{lbl}  [{prof_rows[lbl]['全體']:.0f}]" if "%" in lbl else f"{lbl}  [{prof_rows[lbl]['全體']:.2f}]"
            for lbl in z.index]
    ax.set_yticks(range(len(z)), ylab, fontsize=10)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(ORDER)), minor=True)
    ax.set_yticks(np.arange(-.5, len(z)), minor=True)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="minor", color="white", lw=2)
    ax.axhline(sum(d == 0 for d in dirs) - 0.5, color=INK, lw=1.2)
    ax.set_title("各類型的國七背景與身心（格內為加權平均；顏色＝與全體差距，以標準差計）", fontsize=12.5,
                 color=INK, loc="left", pad=34, fontweight="bold")
    ax.text(0, -0.03, "上方灰色四列（性別、補習、數位區、私校）不涉好壞，深淺只表示與全體差距大小；"
            "\n其餘各列橘＝較不利、藍＝較有利（±1 SD 封頂）。[ ] 內為全體平均；上床時間 23.5 = 23:30。\n"
            "† 遊戲、影音、聊天、瀏覽四種用途的平日時數加總，用途可重疊，會高估實際上網時數，僅作相對比較。",
            transform=ax.transAxes, fontsize=8.8, color=MUTED, va="top", linespacing=1.6)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_type_profiles_w1.png", dpi=200)
    plt.close(fig)


fig_profile()

# 樣本流失：各類型串接到國八的比例
panel = D.load_panel()
A["linked"] = A["nstudent_id"].isin(panel["nstudent_id"]).astype(float)
ret = A.groupby("type", observed=True)["linked"].mean()
say("\n各類型國八追蹤成功率 %：", (100 * ret).round(1).to_dict())

# ---------- 4. 類型 → 國八結果 ----------
KEEP = ["nstudent_id", "type", "type_alt", "maxpost", "straight", "randA", "late_screen_any", "late_screen_main",
        "awake23"]
P = panel.merge(A_all[KEEP], on="nstudent_id", how="inner")
P["female"] = P["female_1"].fillna(P["female_2"])
P["private"] = (P["school_type"] == 2).astype(float)
P["tc"] = P["type"].cat.codes   # 0 = REF（ORDER[0]）
P["tc_alt"] = P["type_alt"].cat.codes
for c in ("bedtime", "sleep_hrs"):
    P[f"{c}_2m"], P[f"{c}_1m"] = P[f"{c}_2"] * 60, P[f"{c}_1"] * 60  # 分鐘
P["selfharm_2pp"] = P["selfharm_2"] * 100
Pm = P[P["randA"]].copy()       # 主分析：課堂 A 卷
OUTC = {  # 結果: (標籤, 前測, 單位, 越大越好?)
    "cesd_nosleep_2": ("國八憂鬱（不含睡眠題）", "cesd_nosleep_1", "分，0–44", False),
    "selfharm_2pp": ("國八自傷／求死意念", "cesd_nosleep_1", "百分點", False),
    "bedtime_2m": ("國八上床時間", "bedtime_1m", "分鐘", False),
    "sleep_hrs_2m": ("國八睡眠時數", "sleep_hrs_1m", "分鐘", True),
    "addict_2": ("國八網路成癮", "addict_1", "分，1–4", False),
    "grade_rank_2": ("國八成績排名", "grade_rank_1", "級距，大=差", False),
    "who5_2": ("國八幸福感 WHO-5", "who5_1", "分，0–20", True),
}
COV = "female + fam_econ_1 + C(digi_level)"
BASECOLS = ["female", "fam_econ_1", "digi_level", "HOUWGT", "nschool_id"]
# M3：再加國七既有身心、學校與家庭、受凌虐（較可能是混淆）；M3b 再加補習、網路成癮、FOMO（可能是中介或類型定義的一部分）
CONF3 = ["who5_1", "life_sat_1", "grade_rank_1", "private", "bully_victim_1", "cyber_victim_1", "cohesion_1",
         "parents_married_cohab_1"]
MED3 = ["cram_1", "addict_1", "fomo_1"]
SPEC_LABEL = {"M1": "M1：性別、家庭經濟、數位區", "M2": "M2：M1＋國七同一指標", "M2s": "M2（M3 樣本）",
              "M3": "M3：M2＋國七身心、學校、家庭、受凌", "M3b": "M3b：M3＋國七補習、網路成癮、FOMO"}
say(f"\n== 4. 類型 → 國八結果（參照：{REF}；WLS/HOUWGT；學校叢集 SE；FDR 跨 7 結果 × {K_FINAL - 1} 對比，各模型分開）==")
say(f"國八串接樣本：課堂 A 卷 n={len(Pm)}（含自願加答 n={len(P)}）")
for k_, v_ in SPEC_LABEL.items():
    say(f"  {k_} = {v_}")
say(f"  M3 加入：{CONF3}；M3b 另加：{MED3}；M2s/M3/M3b 用同一完整樣本（自傷意念的前測為國七憂鬱）")


def spec_terms(base, spec):
    c3 = [c for c in CONF3 if c != base]
    c3b = c3 + [c for c in MED3 if c != base]
    rhs = {"M1": [], "M2": [base], "M2s": [base], "M3": [base] + c3, "M3b": [base] + c3b}[spec]
    need = rhs + (c3b if spec in ("M2s", "M3", "M3b") else [])
    return rhs, list(dict.fromkeys(need))


def fit_outcomes(Pd, tag, tvar="tc", order=None, specs=("M1", "M2")):
    order = order or ORDER
    late = [order.index(t) for t in LATE_ALL if t in order]
    out = []
    for y, (lbl, base, unit, good) in OUTC.items():
        for spec in specs:
            rhs, need = spec_terms(base, spec)
            d = Pd[[y, tvar] + BASECOLS + need].dropna()
            f = wls_cluster(f"{y} ~ C({tvar}){''.join(' + ' + c for c in rhs)} + {COV}", d)
            terms = [p for p in f.params.index if p.startswith(f"C({tvar})")]
            p_omni = wald_p(f, [{p: 1} for p in terms])
            lt = [f"C({tvar})[T.{c}]" for c in late]
            p_eq = (wald_p(f, [{lt[0]: 1, lt[i]: -1} for i in range(1, len(lt))])
                    if len(lt) >= 2 and all(p in terms for p in lt) else np.nan)
            ci = f.conf_int()
            sd_y = np.sqrt(np.cov(d[y], aweights=d["HOUWGT"]))  # 結果的加權 SD（標準化效果量用）
            for p in terms:
                c = int(p.split("[T.")[1].rstrip("]"))
                out.append(dict(sample=tag, outcome=y, label=lbl, unit=unit, higher_better=good, spec=spec,
                                type=order[c], n=int(f.nobs), n_type=int((d[tvar] == c).sum()),
                                est=f.params[p], ci_lo=ci.loc[p, 0], ci_hi=ci.loc[p, 1], p=f.pvalues[p],
                                est_sd=f.params[p] / sd_y, sd_y=sd_y, p_omnibus=p_omni, p_equal_late=p_eq,
                                r2=f.rsquared))
    res = pd.DataFrame(out)
    for spec in specs:
        m = res["spec"] == spec
        res.loc[m, "q_fdr"] = multipletests(res.loc[m, "p"], method="fdr_bh")[1]
    return res


est = fit_outcomes(Pm, "main", specs=("M1", "M2", "M2s", "M3", "M3b"))
sens = fit_outcomes(Pm[(~Pm["straight"]) & (Pm["maxpost"] >= .8)], "sens_certain")   # 排除直線作答與歸類不確定
sens_all = fit_outcomes(P, "with_volunteers")                                         # 含自願加答者
est_alt = fit_outcomes(Pm, f"k{K_ALT}", tvar="tc_alt", order=ORDER_ALT)              # k=7 解
pd.concat([est, sens, sens_all]).round(5).to_csv(TAB / "w2_outcomes_by_type.csv", index=False, encoding="utf-8-sig")
est_alt.round(5).to_csv(TAB / f"w2_outcomes_by_type_k{K_ALT}.csv", index=False, encoding="utf-8-sig")


def show(res, spec, order):
    sh = res[res["spec"] == spec].assign(
        est_ci=lambda d: d.apply(lambda r: f"{r.est:+.2f} [{r.ci_lo:+.2f},{r.ci_hi:+.2f}] q={r.q_fdr:.3f}", axis=1))
    return sh.pivot(index="label", columns="type", values="est_ci")[[t for t in order[1:] if t in set(sh.type)]].T


for spec in ("M1", "M2", "M3"):
    say(f"\n-- {spec} 估計 [95% CI] q（相對 {REF}）--\n" + show(est, spec, ORDER).to_string())
say("\nM2 各結果 omnibus p：", est[est.spec == "M2"].groupby("label")["p_omnibus"].first().round(4).to_dict())
say(f"M2 晚睡型（{'、'.join(LATE)}）係數相等檢定 p：",
    est[est.spec == "M2"].groupby("label")["p_equal_late"].first().round(3).to_dict())
for spec in ("M1", "M2", "M3"):
    say(f"{spec} FDR q<.05 的對比：")
    for _, r in est[(est["spec"] == spec) & (est["q_fdr"] < .05)].iterrows():
        say(f"  {r.type} × {r.label}: {r.est:+.2f}（{r.unit}）[{r.ci_lo:+.2f}, {r.ci_hi:+.2f}], "
            f"p={r.p:.4f}, q={r.q_fdr:.3f}, n={r.n}（該類 {r.n_type}）")
nr = {s: (int(est[est.spec == s]["n"].min()), int(est[est.spec == s]["n"].max())) for s in ("M1", "M2", "M3")}
say("各模型 n 範圍：", nr)


def compare(other, tag):
    cmp = est[est.spec == "M2"].merge(other[other.spec == "M2"], on=["outcome", "type"], suffixes=("", "_o"))
    lost = cmp[(cmp["q_fdr"] < .05) & (cmp["p_o"] >= .05)]
    msg = (f"{tag}：M2 估計值與主分析相關 r={np.corrcoef(cmp['est'], cmp['est_o'])[0, 1]:.3f}；主分析 q<.05 者仍 p<.05："
           f"{((cmp['q_fdr'] < .05) & (cmp['p_o'] < .05)).sum()}/{(cmp['q_fdr'] < .05).sum()}"
           + ("；不再顯著：" + "、".join(f"{r.type}×{r.label}（p={r.p_o:.2f}）" for _, r in lost.iterrows())
              if len(lost) else ""))
    say(msg)
    return msg


say("\n== 敏感度 ==")
msg_c = compare(sens, f"排除直線作答且最大後驗 <.8（n≈{int(sens['n'].max())}）")
msg_v = compare(sens_all, f"含事後自願加答者（n≈{int(sens_all['n'].max())}）")
m23 = est[est.spec.isin(["M2s", "M3", "M3b"])].pivot_table(index=["label", "type"], columns="spec", values="est")
say("M2s → M3 → M3b（同一樣本）估計值：\n" + m23.loc[[l for l in m23.index if l[1] in LATE]].round(2).to_string())
say(f"\n-- k={K_ALT} 解 M2 估計 [95% CI] q（相對 {ORDER_ALT[0]}）--\n" + show(est_alt, "M2", ORDER_ALT).to_string())

# 晚睡型合併（避免以「誰過了 p<.05」區分各型）
codes_l = [ORDER.index(t) for t in LATE]
POOL = f"{'、'.join(LATE)}（合併）"
pool = fit_outcomes(Pm.assign(tp=Pm["tc"].where(~Pm["tc"].isin(codes_l), codes_l[0])), "pooled_late", tvar="tp",
                    specs=("M1", "M2", "M3"))
pool = pool[pool["type"] == ORDER[codes_l[0]]].assign(type=POOL).drop(columns=["p_equal_late"])
for spec in ("M1", "M2", "M3"):
    m = pool["spec"] == spec
    pool.loc[m, "q_fdr"] = multipletests(pool.loc[m, "p"], method="fdr_bh")[1]
pool = pool.merge(est[est.spec.isin(["M1", "M2", "M3"])].groupby(["outcome", "spec"])["p_equal_late"].first()
                  .reset_index(), on=["outcome", "spec"])
pool.round(5).to_csv(TAB / "w2_late_types_pooled.csv", index=False, encoding="utf-8-sig")
say(f"\n{POOL} vs {REF}（q 為 7 結果內 FDR；p_equal = 各型相等檢定）：")
for _, r in pool[pool.spec != "M1"].iterrows():
    say(f"  {r.spec} {r.label}: {r.est:+.2f}（{r.unit}）[{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] p={r.p:.4f} q={r.q_fdr:.3f}"
        f"；各型相等 p={r.p_equal_late:.2f}")

# ---------- 4b. 深夜本身 vs 螢幕 ----------
say("\n== 4b. 深夜本身 vs 螢幕 ==")
# (1) 巢套：M2 再加國七自填上床時間，類型是否仍有額外預測力
nest = []
for y, (lbl, base, unit, good) in OUTC.items():
    if base == "bedtime_1m":
        continue
    d = Pm[[y, "tc", base, "bedtime_1"] + BASECOLS].dropna()
    f2 = wls_cluster(f"{y} ~ C(tc) + {base} + {COV}", d)
    fb = wls_cluster(f"{y} ~ bedtime_1 + {base} + {COV}", d)
    fbt = wls_cluster(f"{y} ~ C(tc) + bedtime_1 + {base} + {COV}", d)
    terms = [p for p in fbt.params.index if p.startswith("C(tc)")]
    row = dict(outcome=y, label=lbl, unit=unit, n=int(fbt.nobs), p_type_M2=wald_p(f2, [{p: 1} for p in terms]),
               p_type_given_bedtime=wald_p(fbt, [{p: 1} for p in terms]),
               R2_M2=f2.rsquared, R2_bedtime=fb.rsquared, R2_both=fbt.rsquared,
               dR2_type_given_bedtime=fbt.rsquared - fb.rsquared, dR2_bedtime_given_type=fbt.rsquared - f2.rsquared,
               bed_per_h=fbt.params["bedtime_1"], bed_lo=fbt.conf_int().loc["bedtime_1", 0],
               bed_hi=fbt.conf_int().loc["bedtime_1", 1], bed_p=fbt.pvalues["bedtime_1"])
    for t in LATE:
        p = f"C(tc)[T.{ORDER.index(t)}]"
        row[f"{t}_M2"], row[f"{t}_M2+bed"], row[f"{t}_M2+bed_p"] = f2.params[p], fbt.params[p], fbt.pvalues[p]
    nest.append(row)
nest = pd.DataFrame(nest)
nest.round(5).to_csv(TAB / "w2_type_vs_bedtime.csv", index=False, encoding="utf-8-sig")
for _, r in nest.iterrows():
    say(f"  {r.label}（n={r.n}）：類型 omnibus p {r.p_type_M2:.3f} → 加上國七上床時間後 {r.p_type_given_bedtime:.3f}；"
        f"R² 上床時間模型 {r.R2_bedtime:.4f} → 再加類型 {r.R2_both:.4f}（ΔR²={r.dR2_type_given_bedtime:.4f}）；"
        f"上床時間每晚 1 小時 {r.bed_per_h:+.2f} [{r.bed_lo:+.2f}, {r.bed_hi:+.2f}] p={r.bed_p:.3f}；"
        + "、".join(f"{t} {r[f'{t}_M2']:+.2f}→{r[f'{t}_M2+bed']:+.2f}" for t in LATE))

# (2) 上床時間與深夜螢幕（23:00–02:00 主要或次要活動含螢幕）同時進入；及 23:00 仍醒者中的深夜螢幕
lvs = []
for y in ("cesd_nosleep_2", "selfharm_2pp", "who5_2", "addict_2"):
    lbl, base, unit, _ = OUTC[y]
    d = Pm[[y, base, "bedtime_1", "late_screen_any", "awake23"] + BASECOLS].dropna()
    for model, rhs, dd in (("僅上床時間", ["bedtime_1"], d), ("僅深夜螢幕", ["late_screen_any"], d),
                           ("兩者同時", ["bedtime_1", "late_screen_any"], d),
                           ("23:00 仍醒者中", ["late_screen_any"], d[d["awake23"] == 1])):
        f = wls_cluster(f"{y} ~ {' + '.join(rhs)} + {base} + {COV}", dd)
        for term in rhs:
            lvs.append(dict(outcome=y, label=lbl, unit=unit, model=model, term=term, n=int(f.nobs),
                            n_exposed=int(dd[term].sum()) if term == "late_screen_any" else np.nan,
                            est=f.params[term], ci_lo=f.conf_int().loc[term, 0], ci_hi=f.conf_int().loc[term, 1],
                            p=f.pvalues[term]))
lvs = pd.DataFrame(lvs)
lvs.round(5).to_csv(TAB / "w2_late_night_vs_screen.csv", index=False, encoding="utf-8-sig")
say("  單位：上床時間 = 每晚 1 小時；深夜螢幕 = 有 vs 無（23:00–02:00 主要或次要活動含螢幕）；皆控制 M2 共變項")
for _, r in lvs.iterrows():
    say(f"  {r.label}｜{r.model}｜{r.term}: {r.est:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] p={r.p:.3f} n={r.n}")

# (3) 補習類依日誌作息拆三組（規則式，取代 k=7 不穩定的「補習後晚睡型」）：
#     23:00 已睡；23:00 仍醒且深夜有螢幕；23:00 仍醒但深夜無螢幕
split = None
if CRAM in ORDER:
    c0 = ORDER.index(CRAM)
    awake, scr = Pm["awake23"] == 1, Pm["late_screen_any"] == 1
    Ps = Pm.assign(tsc=np.select([(Pm["tc"] == c0) & awake & scr, (Pm["tc"] == c0) & awake & ~scr],
                                 [K_FINAL, K_FINAL + 1], Pm["tc"]))
    Ps = Ps.assign(tsa=np.where(Ps["tsc"] == K_FINAL + 1, K_FINAL, Ps["tsc"]))   # 兩個仍醒組合併
    SPL = {c0: f"{CRAM}：23:00 已睡", K_FINAL: f"{CRAM}：23:00 仍醒＋深夜螢幕", K_FINAL + 1: f"{CRAM}：23:00 仍醒、無深夜螢幕"}
    o3 = [SPL.get(i, t) for i, t in enumerate(ORDER)] + [SPL[K_FINAL], SPL[K_FINAL + 1]]
    o2 = [SPL.get(i, t) for i, t in enumerate(ORDER)] + [f"{CRAM}：23:00 仍醒（合併）"]
    s3 = fit_outcomes(Ps, "cram_split3", tvar="tsc", order=o3, specs=("M2",))
    s2 = fit_outcomes(Ps, "cram_split2", tvar="tsa", order=o2, specs=("M2",))
    split = pd.concat([s3[s3["type"].str.startswith(CRAM)], s2[s2["type"] == o2[-1]]]).drop(
        columns=["q_fdr", "p_equal_late"])
    pdiff = {}
    for y, (lbl, base, _, _) in OUTC.items():
        d = Ps[[y, "tsc", "tsa", base] + BASECOLS].dropna()
        f3 = wls_cluster(f"{y} ~ C(tsc) + {base} + {COV}", d)
        f2_ = wls_cluster(f"{y} ~ C(tsa) + {base} + {COV}", d)
        pdiff[y] = dict(p_screen_vs_noscreen=wald_p(f3, [{f"C(tsc)[T.{K_FINAL}]": 1, f"C(tsc)[T.{K_FINAL + 1}]": -1}]),
                        p_awake_vs_asleep=wald_p(f2_, [{f"C(tsa)[T.{K_FINAL}]": 1}] if c0 == 0 else
                                                 [{f"C(tsa)[T.{K_FINAL}]": 1, f"C(tsa)[T.{c0}]": -1}]))
    split = split.join(pd.DataFrame(pdiff).T, on="outcome")
    split.round(5).to_csv(TAB / "w2_cram_split_by_late_night.csv", index=False, encoding="utf-8-sig")
    say(f"\n{CRAM}依日誌作息拆分（M2，相對 {REF}；p_awake_vs_asleep＝仍醒 vs 已睡；p_screen＝仍醒者中有 vs 無深夜螢幕）：")
    for _, r in split.iterrows():
        say(f"  {r.type} × {r.label}: {r.est:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] p={r.p:.3f}（該組 {r.n_type}）"
            f"；仍醒 vs 已睡 p={r.p_awake_vs_asleep:.3f}；有 vs 無螢幕 p={r.p_screen_vs_noscreen:.2f}")

# 國八描述（加權平均，未調整）
desc = {}
for t in ORDER:
    m, ww = Pm["type"] == t, Pm.loc[Pm["type"] == t, "HOUWGT"]
    flag = lambda s, c: (s >= c).astype(float).where(s.notna())  # noqa: E731
    desc[t] = {"n": int(m.sum()),
               "國八 CESD≥16 %": 100 * wmean(flag(Pm.loc[m, "cesd_2"], 16), ww),
               "國八自傷/求死意念 %": 100 * wmean(Pm.loc[m, "selfharm_2"], ww),
               "國七午夜後才睡 %": 100 * wmean(flag(Pm.loc[m, "bedtime_1"], 24), ww),
               "國八午夜後才睡 %": 100 * wmean(flag(Pm.loc[m, "bedtime_2"], 24), ww),
               "國八睡眠時數": wmean(Pm.loc[m, "sleep_hrs_2"], ww),
               "國八網路成癮": wmean(Pm.loc[m, "addict_2"], ww),
               "國八 WHO-5": wmean(Pm.loc[m, "who5_2"], ww)}
desc = pd.DataFrame(desc).T
say("\n== 國八描述（加權、未調整）==\n" + desc.round(1).to_string())
desc.round(3).to_csv(TAB / "w2_descriptives_by_type.csv", encoding="utf-8-sig")


def fig_forest():
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 7.4), sharey=True)
    types = ORDER[1:]
    yy = np.arange(len(types))[::-1]
    SP = (("M1", 0.25, "D", "#9b9a95", 1.1), ("M2", 0.0, "o", "#2a78d6", 1.9), ("M3", -0.25, "s", "#e3592f", 1.1))
    for ax, (y, (lbl, base, unit, good)) in zip(axes.flat, OUTC.items()):
        for spec, off, mk, col, lw in SP:
            d = est[(est.outcome == y) & (est.spec == spec)].set_index("type").reindex(types)
            ax.hlines(yy + off, d.ci_lo, d.ci_hi, color=col, lw=lw)
            sig = (d.q_fdr < .05).to_numpy()
            ms = 40 if spec == "M2" else 24
            ax.scatter(d.est[~sig], yy[~sig] + off, marker=mk, s=ms, facecolor="white", edgecolor=col, lw=1.3, zorder=3)
            ax.scatter(d.est[sig], yy[sig] + off, marker=mk, s=ms + 4, color=col, edgecolor="white", lw=0.8, zorder=3)
            if spec == "M2":
                for yi, (_, r) in zip(yy, d.iterrows()):
                    small = r.n_type < 100
                    if r.q_fdr < .05 or small:
                        pos = r.est >= 0
                        txt = (f"{r.est:+.1f}" if abs(r.est) >= 1 else f"{r.est:+.2f}") if r.q_fdr < .05 else ""
                        ax.annotate(txt + ("†" if small else ""), (r.ci_hi if pos else r.ci_lo, yi + off),
                                    xytext=(4 if pos else -4, 0), textcoords="offset points",
                                    ha="left" if pos else "right", va="center", fontsize=8.5, color=INK)
        ax.axvline(0, color=MUTED, lw=0.9)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        note = "\n前測＝國七憂鬱（國七無自傷題）" if y == "selfharm_2pp" else ""
        ax.set_title(lbl + note, fontsize=11.5 if not note else 10.5, loc="left", fontweight="bold", color=INK)
        ax.set_xlabel(f"與{REF}之差（{unit}；{'正值較好' if good else '正值較差'}）", fontsize=8.8, color=MUTED)
        ax.set_yticks(yy, types, fontsize=10)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo - (hi - lo) * 0.08, hi + (hi - lo) * 0.12)
    lax = axes.flat[-1]
    lax.axis("off")
    handles = [Line2D([], [], color="#2a78d6", marker="o", lw=1.9, ms=7, label="M2：再控制國七同一指標\n（主要結果）"),
               Line2D([], [], color="#9b9a95", marker="D", lw=1.1, ms=5, label="M1：僅控制性別、家庭經濟、\n數位發展區"),
               Line2D([], [], color="#e3592f", marker="s", lw=1.1, ms=5,
                      label="M3：M2＋國七幸福感、生活滿意、\n成績、私校、受凌、家庭"),
               Line2D([], [], color=MUTED, marker="o", lw=0, ms=7, markerfacecolor="white",
                      label="空心＝FDR 校正後不顯著\n實心＝q < .05（數字為 M2 估計）")]
    lax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9.2, labelspacing=1.1, borderaxespad=0)
    lax.text(0, 0.0, f"點與線：估計值與 95% CI；加權 WLS、學校叢集標準誤\n"
             f"國七課堂 A 卷且追蹤到國八；M2 n={nr['M2'][0]:,}–{nr['M2'][1]:,}，\nM3 n={nr['M3'][0]:,}–{nr['M3'][1]:,}"
             "（M3 完整樣本較小）\n† 該類有效樣本 <100，估計不穩", transform=lax.transAxes, fontsize=8.5,
             color=MUTED, linespacing=1.5)
    fig.suptitle(f"國七的晚上類型，一年後怎麼樣？（相對於「{REF}」）", fontsize=16, fontweight="bold",
                 x=0.01, ha="left", y=0.99, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.0)
    fig.savefig(FIG / "fig2_w2_outcomes_forest.png", dpi=200)
    plt.close(fig)


def fig_ksel():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))
    ax = axes[0]
    for c, lbl, col, mk in (("ASW_LCA", "LCA", "#2a78d6", "o"), ("ASW_PAM", "PAM-DHD", "#e3592f", "s"),
                            ("ASW_Ward", "Ward-DHD", "#1baf7a", "^"), ("ASW_kmodes", "k-modes", "#a8326e", "D")):
        ax.plot(KS, ksel[c], marker=mk, color=col, lw=2, ms=6, label=lbl)
    ax.axhline(.25, color=MUTED, lw=0.8, ls=":")
    ax.text(KS[0], .252, "0.25（低於此＝無明顯結構）", fontsize=8, color=MUTED, va="bottom")
    ax.set_title("平均輪廓係數 ASW（DHD 距離）", fontsize=11.5, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower left")
    ax = axes[1]
    ax.plot(KS, ksel["LCA_BIC"], marker="o", color="#2a78d6", lw=2, ms=6)
    ax.set_title("LCA 的 BIC（越低越好；數字＝比前一個 k 降低多少）", fontsize=11.5, loc="left", fontweight="bold")
    for k, d in zip(KS[1:], -np.diff(ksel["LCA_BIC"])):
        ax.annotate(f"−{d:,.0f}", (k, ksel.loc[k, "LCA_BIC"]), xytext=(5, 7), textcoords="offset points",
                    fontsize=8.5, color=MUTED)
    ax = axes[2]
    for m, lbl, col, mk in (("lca", "LCA", "#2a78d6", "o"), ("pam", "PAM-DHD", "#e3592f", "s")):
        ax.plot(KS, ksel[f"boot_ARI_{m}"], marker=mk, color=col, lw=2, ms=6, label=f"{lbl} 平均 ARI")
        ax.plot(KS, ksel[f"boot_minJaccard_{m}"], marker=mk, color=col, lw=1.2, ls="--", ms=5,
                markerfacecolor="white", label=f"{lbl} 最不穩一類 Jaccard")
    ax.axhline(.75, color=MUTED, lw=0.8, ls=":")
    ax.axhline(.5, color=MUTED, lw=0.8, ls=":")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"學校叢集 bootstrap（B={N_BOOT}，從頭重擬）", fontsize=11.5, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8.2, loc="lower left")
    for ax in axes:
        ax.axvline(K_ALT, color="#9b9a95", lw=6, alpha=.30, zorder=0)
        ax.axvline(K_FINAL, color="#f2b134", lw=6, alpha=.40, zorder=0)
        ax.set_xticks(KS)
        ax.set_xlabel("類別數 k", fontsize=9.5)
        ax.grid(axis="y", color=GRID)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(f"選擇類別數：採 k={K_FINAL}（黃帶）。與 k={K_ALT}（灰帶）相比，{len(fin_better)}/{len(CMP_IDX)} 項輪廓係數與"
                 f"穩定度指標以 k={K_FINAL} 較佳；BIC 持續下降、k={K_FINAL} 之後降幅趨緩",
                 fontsize=12.5, fontweight="bold", x=0.01, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "fig_s1_k_selection.png", dpi=200)
    plt.close(fig)


fig_forest()
fig_ksel()

# ---------- 5. results.md ----------
kf, kalt = ksel.loc[K_FINAL], ksel.loc[K_ALT]
dbic = (-ksel["LCA_BIC"].diff()).dropna().round(0).astype(int).to_dict()
asw_rng = (ksel[ASW_COLS].min().min(), ksel[ASW_COLS].max().max())
e2 = est[est.spec == "M2"].set_index(["outcome", "type"])
e3 = est[est.spec == "M3"].set_index(["outcome", "type"])
ea = est_alt[est_alt.spec == "M2"].set_index(["outcome", "type"])
pl = pool.set_index(["outcome", "spec"])
nb = nest.set_index("outcome")
lv = lvs.set_index(["outcome", "model", "term"])
sp_ = split.set_index(["outcome", "type"]) if split is not None else None


def fmt(r, sd=True):
    return (f"{r.est:+.2f} {r.unit.split('，')[0]}（{'≈%+.2f SD；' % r.est_sd if sd else ''}95% CI {r.ci_lo:+.2f}～"
            f"{r.ci_hi:+.2f}；p={r.p:.2g}；q={r.q_fdr:.2g}；n={r.n}，該類 {r.n_type}）")


def pf(p):
    return "p<.001" if p < .001 else f"p={p:.3f}".replace("0.", ".")


def ci_txt(r, unit=""):
    return f"{r.est:+.2f}{unit} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]，{pf(r.p)}"


own = [t for t in ORDER if rep.loc[t, "own"]]
noown = [t for t in ORDER if not rep.loc[t, "own"]]
pam_merge = {}
for t in ORDER:
    for t2 in rep.loc[t, "shared"]:
        key = tuple(sorted([t, t2], key=ORDER.index))
        pam_merge[key] = tuple(xt.loc[u, rep.loc[u, "pam"]] for u in key)
eq_same = [OUTC[y][0] for y in OUTC if pl.loc[(y, "M2"), "p_equal_late"] >= .05]
eq_diff = [f"{OUTC[y][0]} p={pl.loc[(y, 'M2'), 'p_equal_late']:.2f}" for y in OUTC if pl.loc[(y, "M2"), "p_equal_late"] < .05]
gone = [r.label for _, r in nest.iterrows() if r.p_type_M2 < .05 <= r.p_type_given_bedtime]
kept = [r.label for _, r in nest.iterrows() if r.p_type_M2 < .05 and r.p_type_given_bedtime < .05]
LC = OUTC["cesd_nosleep_2"][0]
summary = [
    "## 重點摘要（由程式自動產生）", "",
    f"- 樣本：W1 A 卷 {int(w1['has_A'].sum())} 人，有效晚間日誌 n={N}（分群用）；其中課堂作答 A 卷者（以學校為單位分派） n={len(A)}"
    f"（占比、輪廓與國八分析用；另 {int((~RA).sum())} 人為課堂填 B 卷後自願加答 A 卷，非隨機，只列敏感度）；"
    f"課堂 A 卷追蹤到國八 n={len(Pm)}。",
    f"- 深夜有多普遍（加權）：日誌中午夜 0:00–1:00 仍醒著 {OVR['awake0']:.1f}%；其中約 "
    f"{OVR['awake0'] - OVR['awake0_ex'] * (100 - sizes.loc[CHAOS, 'pct_w']) / 100:.1f} 個百分點來自作答品質可疑的{CHAOS}，"
    f"排除後為 {OVR['awake0_ex']:.1f}%；同一樣本自填上床時間在午夜或更晚者 {OVR['self_bed']:.1f}%。"
    f"23:00–02:00 任一時段有螢幕活動：只算主要活動 {OVR['late_main']:.1f}%，連同次要活動 {OVR['late_any']:.1f}%。",
    f"- 選 k：採 LCA k={K_FINAL}。與 k={K_ALT} 相比，{len(fin_better)}/{len(CMP_IDX)} 項指標（四法輪廓係數、LCA bootstrap "
    f"平均 ARI 與最不穩一類 Jaccard）以 k={K_FINAL} 較佳；各指標最佳 k：{best_k}；BIC 一路下降（每多一類的降幅 {dbic}，"
    f"k={K_FINAL} 之後趨緩）。k={K_FINAL} 的品質：entropy R²={kf.LCA_entropyR2:.2f}；從頭重擬的學校叢集 bootstrap 平均 "
    f"ARI={kf.boot_ARI_lca:.2f}（第 5 百分位 {kf.boot_ARI_lca_p05:.2f}），各類 Jaccard {jac_final.min():.2f}–"
    f"{jac_final.max():.2f}（最低為 {jac_final.idxmin()}）。"
    + (f"k={K_ALT} 會多分出「{CL}」（補習後 23:00 仍醒），但它在 bootstrap 中平均 Jaccard 只有 {CL_STAB['mean']:.2f}，"
       f"{CL_STAB['lt50']:.0%} 的重抽樣本 <0.5（Hennig 2007 視為「解散」），多數重抽樣本中這群人併回補習類"
       f"（在 k={K_FINAL} 中 {CL_STAB['to_main_pct']:.0%} 屬「{CL_STAB['to_main']}」）。因此不把它當成一個類型，"
       f"改以規則式切分（{CRAM}中 23:00 是否仍醒、深夜是否用螢幕）分析，k={K_ALT} 結果列為敏感度。" if CL_STAB else ""),
    "- 類型（加權占比 [95% CI]；午夜 0:00–1:00 仍醒著的比例）：" + "；".join(
        f"{t} {r.pct_w:.1f}% [{r.ci_lo:.1f}, {r.ci_hi:.1f}]，午夜仍醒 {r.awake_midnight_pct:.0f}%" for t, r in sizes.iterrows())
    + f"。兩種螢幕晚睡型（{'、'.join(LATE)}）合計 {sizes.loc[LATE, 'pct_w'].sum():.1f}%，但各型只有約三分之一午夜仍醒"
    f"（{sizes.loc[LATE, 'awake_midnight_pct'].min():.0f}–{sizes.loc[LATE, 'awake_midnight_pct'].max():.0f}%），"
    f"23:00 時段已有 {sizes.loc[LATE, 'sleep_at_23_pct'].min():.0f}–{sizes.loc[LATE, 'sleep_at_23_pct'].max():.0f}% 入睡；"
    "名稱中的「夜貓」是相對於其他類型而言。",
    f"- 深夜螢幕（23:00–02:00，主要或次要活動）在各型的比例：" + "；".join(
        f"{t} {r.late_screen_any_pct:.0f}%（只算主要 {r.late_screen_main_pct:.0f}%）" for t, r in sizes.iterrows()) + "。"
    + (f"{CRAM}中 23:00 仍醒者占 {OVR['cram_awake']:.0f}%（n={OVR['cram_awake_n']}，平均自填上床 "
       f"{OVR['cram_awake_bed']:.2f} 時），其中 {OVR['cram_awake_scr']:.0f}% 深夜有螢幕，其餘是讀書或生活作息。"
       if "cram_awake" in OVR else ""),
    "- 國八描述（加權、未調整；CESD≥16 %／自傷求死意念 %／國七→國八午夜後才睡 %）：" + "；".join(
        f"{t} {r['國八 CESD≥16 %']:.0f}／{r['國八自傷/求死意念 %']:.0f}／{r['國七午夜後才睡 %']:.0f}→{r['國八午夜後才睡 %']:.0f}"
        for t, r in desc.iterrows()),
    "- 國八結果，M2（控制國七同一指標，參照「" + REF + "」）FDR q<.05：", "",
    *[f"  - {r.label}：{r.type} {fmt(r)}" + ("（該類有效樣本 <100，不穩）" if r.n_type < 100 else "")
      for _, r in est[(est.spec == "M2") & (est.q_fdr < .05)].iterrows()], "",
    f"- 國八 FDR q<.05 的對比數：M1 {int((est[est.spec == 'M1'].q_fdr < .05).sum())}/{int((est.spec == 'M1').sum())}；"
    f"M2 {int((est[est.spec == 'M2'].q_fdr < .05).sum())}/{int((est.spec == 'M2').sum())}；"
    f"M3 {int((est[est.spec == 'M3'].q_fdr < .05).sum())}/{int((est.spec == 'M3').sum())}。"
    f"各模型 n：M1 {nr['M1'][0]:,}–{nr['M1'][1]:,}；M2 {nr['M2'][0]:,}–{nr['M2'][1]:,}；M3 {nr['M3'][0]:,}–{nr['M3'][1]:,}。",
    f"- {'、'.join(LATE)}應合併陳述，不宜比較誰「過了 p<.05」：M2 兩型係數相等檢定在 {'、'.join(eq_same)} 皆 p≥.05"
    + (f"（例外：{'、'.join(eq_diff)}）" if eq_diff else "") + "。合併後（M2，參照 " + REF + "）：憂鬱 "
    f"{ci_txt(pl.loc[('cesd_nosleep_2', 'M2')], ' 分')}（q{pf(pl.loc[('cesd_nosleep_2', 'M2'), 'q_fdr'])[1:]}）；自傷／求死意念 "
    f"{ci_txt(pl.loc[('selfharm_2pp', 'M2')], ' 百分點')}（q{pf(pl.loc[('selfharm_2pp', 'M2'), 'q_fdr'])[1:]}）；WHO-5 "
    f"{ci_txt(pl.loc[('who5_2', 'M2')], ' 分')}；上床時間 {ci_txt(pl.loc[('bedtime_2m', 'M2')], ' 分鐘')}；睡眠 "
    f"{ci_txt(pl.loc[('sleep_hrs_2m', 'M2')], ' 分鐘')}；網路成癮 {ci_txt(pl.loc[('addict_2', 'M2')], ' 分')}。"
    "睡眠差距是「國七睡眠相同者之間，國八時比參照型少睡」的組間條件差異，不是個人睡眠減少。",
    f"- 更多控制（M3；同一完整樣本 n={int(est[(est.spec == 'M3') & (est.outcome == 'cesd_nosleep_2')]['n'].iloc[0]):,}）："
    "憂鬱差距 " + "；".join(
        f"{t} M2 {m23.loc[(LC, t), 'M2s']:+.2f} → M3 {m23.loc[(LC, t), 'M3']:+.2f}"
        f"（{pf(e3.loc[('cesd_nosleep_2', t), 'p'])}）→ M3b {m23.loc[(LC, t), 'M3b']:+.2f}" for t in LATE)
    + f"；合併 M3 {ci_txt(pl.loc[('cesd_nosleep_2', 'M3')], ' 分')}；自傷意念合併 M3 "
    f"{ci_txt(pl.loc[('selfharm_2pp', 'M3')], ' 百分點')}。M2 的差距應視為上限。",
    f"- 是「深夜」還是「螢幕」？(1) 巢套比較：M2 再加國七自填上床時間後，類型的整體檢定在 {'、'.join(gone)} 不再顯著"
    + (f"（仍顯著：{'、'.join(kept)}）" if kept else "") + f"。以憂鬱為例，p 由 {pf(nb.loc['cesd_nosleep_2', 'p_type_M2'])} "
    f"變為 {pf(nb.loc['cesd_nosleep_2', 'p_type_given_bedtime'])}，R² 只增加 {nb.loc['cesd_nosleep_2', 'dR2_type_given_bedtime']:.4f}；"
    f"上床時間每晚 1 小時 {nb.loc['cesd_nosleep_2', 'bed_per_h']:+.2f} 分（{pf(nb.loc['cesd_nosleep_2', 'bed_p'])}）。"
    "類型對身心結果的預測力大多可由「幾點睡」解釋，晚上「做什麼」（遊戲或影音社群）不應解讀為內容效果。"
    f"(2) 上床時間與深夜螢幕同時放入：憂鬱 上床時間 {ci_txt(lv.loc[('cesd_nosleep_2', '兩者同時', 'bedtime_1')], ' 分／時')}，"
    f"深夜螢幕 {ci_txt(lv.loc[('cesd_nosleep_2', '兩者同時', 'late_screen_any')], ' 分')}；WHO-5 上床時間 "
    f"{ci_txt(lv.loc[('who5_2', '兩者同時', 'bedtime_1')], ' 分／時')}，深夜螢幕 "
    f"{ci_txt(lv.loc[('who5_2', '兩者同時', 'late_screen_any')], ' 分')}；自傷／求死意念 深夜螢幕 "
    f"{ci_txt(lv.loc[('selfharm_2pp', '兩者同時', 'late_screen_any')], ' 百分點')}。23:00 仍醒者"
    f"（n={int(lv.loc[('cesd_nosleep_2', '23:00 仍醒者中', 'late_screen_any'), 'n'])}）中，深夜有無螢幕對憂鬱 "
    f"{ci_txt(lv.loc[('cesd_nosleep_2', '23:00 仍醒者中', 'late_screen_any')], ' 分')}。訊號主要跟著「晚睡」走；"
    "深夜螢幕在晚睡之外沒有可偵測的額外關聯，但信賴區間寬，不能排除中等大小的螢幕效果；深夜醒著多半伴隨螢幕，兩者難以完全分開。",
] + ([f"- {CRAM}依日誌作息拆分（規則式；M2 憂鬱，相對 {REF}）：" + "；".join(
    f"{t.split('：')[1]} {ci_txt(sp_.loc[('cesd_nosleep_2', t)], ' 分')}（{sp_.loc[('cesd_nosleep_2', t), 'n_type']} 人）"
    for t in sp_.loc['cesd_nosleep_2'].index)
    + f"；仍醒 vs 已睡 {pf(sp_.loc['cesd_nosleep_2']['p_awake_vs_asleep'].iloc[0])}，仍醒者中有 vs 無深夜螢幕 "
    f"{pf(sp_.loc['cesd_nosleep_2']['p_screen_vs_noscreen'].iloc[0])}。WHO-5：" + "；".join(
    f"{t.split('：')[1]} {ci_txt(sp_.loc[('who5_2', t)], ' 分')}" for t in sp_.loc['who5_2'].index)
    + f"。補習後 23:00 仍醒者（合併）的方向與螢幕晚睡型一致（螢幕兩型合併：憂鬱 {pl.loc[('cesd_nosleep_2', 'M2'), 'est']:+.2f}、"
    f"WHO-5 {pl.loc[('who5_2', 'M2'), 'est']:+.2f}）；仍醒者中有無深夜螢幕的差異不顯著，但子組只有 "
    f"{int(sp_.loc['cesd_nosleep_2']['n_type'].iloc[1:3].min())}–{int(sp_.loc['cesd_nosleep_2']['n_type'].iloc[1:3].max())} 人，"
    "無法判斷螢幕是否額外有害，只能當輔助證據。"] if sp_ is not None else []) + [
    f"- k={K_ALT} 敏感度（M2 憂鬱，相對 {ORDER_ALT[0]}）：" + "；".join(
        f"{t} {ci_txt(ea.loc[('cesd_nosleep_2', t)], ' 分')}" for t in ORDER_ALT[1:] if ('cesd_nosleep_2', t) in ea.index)
    + "。螢幕兩型（k=6 → k=7）：" + "；".join(
        f"{t} {e2.loc[('cesd_nosleep_2', t), 'est']:+.2f} → {ea.loc[('cesd_nosleep_2', t), 'est']:+.2f}" for t in LATE
        if ('cesd_nosleep_2', t) in ea.index) + "，不受 k 影響。",
    f"- 其他敏感度：{msg_c}；{msg_v}。", "",
    "## 方法決策與限制", "",
    f"- 狀態：21 種活動 → D.DIARY_GROUPS 10 類 → 7 狀態。運動社團才藝、與家人朋友聊天、其他三小類各時段最大占比 "
    f"{small_max}（多數 <3%），併入「生活與其他」；三種螢幕活動分開保留。「螢幕：遊戲／電視」對應題目代碼 14"
    "（看電視／玩遊戲機／網路遊戲／手遊），無法區分電視與遊戲。",
    "- 日誌每時段另有「次要活動」複選（as70b、d、f … p，各 21 項，0/1）。分群只用主要活動（每格一個狀態）；"
    "螢幕暴露另以「主要或次要活動含任何螢幕（代碼 14–19）」建構（主圖折線、type_sizes 的 late_screen_any_pct）。"
    "多數學生也把主要活動勾在次要欄，次要欄宜視為「該時段所有活動」。",
    "- 遺漏：全空白者排除；「睡覺」後的空白視為仍在睡；其餘 ≤2 格以相鄰時段補值（多為 18:00 時段）；仍 >2 格者排除。",
    f"- 樣本：分群用全部有效日誌（n={N}，以人為單位、不加權）。占比、輪廓與國八分析只用課堂作答 A 卷者（以學校為單位分派）"
    f"（response 1、3；n={len(A)}），使用 HOUWGT 與學校叢集 SE。response 2（課堂作答 B 卷後自願加答 A 卷，"
    "截止至 9/15）與 5 非隨機；其中回家讀書型比例偏高（見執行紀錄），已排除於主分析並另作敏感度。",
    f"- 這些類型是「連續光譜的精簡切分」，不是自然分群：所有方法與 k 的 ASW 介於 {asw_rng[0]:.2f}–{asw_rng[1]:.2f}"
    "（<0.25，依 Kaufman & Rousseeuw 屬「無明顯結構」）。",
    f"- 為何以 LCA 為主解：有 BIC／entropy 與後驗機率（可做歸類不確定性的敏感度）；k={K_FINAL} 時 LCA 解的 ASW="
    f"{kf.ASW_LCA:.2f}，與 PAM-DHD（{kf.ASW_PAM:.2f}）相近；同樣從頭重新擬合的 bootstrap 下，LCA 平均 ARI "
    f"{kf.boot_ARI_lca:.2f}、最不穩一類 Jaccard {kf.boot_minJaccard_lca:.2f}，PAM 為 {kf.boot_ARI_pam:.2f}、"
    f"{kf.boot_minJaccard_pam:.2f}。跨方法一致性有限（k={K_FINAL} 時 LCA 與 PAM／Ward／k-modes 的 ARI "
    f"{kf.ARI_LCA_PAM:.2f}／{kf.ARI_LCA_Ward:.2f}／{kf.ARI_LCA_kmodes:.2f}）：在 PAM 有自己對應群（≥70% 落在同一群且不與"
    f"其他類型共用）的有 {'、'.join(own) if own else '（無）'}；沒有的有 {'、'.join(noown) if noown else '（無）'}"
    + ("（" + "；".join(f"{a}與{b}在 PAM 落入同一群，{sa:.0%}／{sb:.0%}" for (a, b), (sa, sb) in pam_merge.items()) + "）"
       if pam_merge else "") + "。沒有 PAM 對應的類型應視為 LCA 特有的切分。",
    f"- LCA 的局部解：k={K_FINAL} 時 {kf.LCA_starts_within1LL} 組起始值的對數概似與最佳解相差 <1，"
    f"其分類與最佳解的 ARI 最低 {kf.LCA_near_best_minARI:.3f}，最佳解可被唯一辨識。bootstrap 每次從頭以 {N_START_BOOT} 組"
    "隨機起始值重新擬合（不以原解暖啟動），與 PAM 的比較條件一致。",
    "- 命名以各類逐時組成的規則自動指派（見 RULES）。螢幕晚睡型名稱中的「夜貓」指相對晚睡：各型只有約三分之一午夜仍醒。"
    + (f"「{CHAOS}」日誌顯示整晚未睡、主要活動為吃飯盥洗家事等，但有填上床時間者 "
       f"{sizes.loc[CHAOS, 'bed_before_midnight_pct']:.0f}% 自填午夜前就寢（上床時間遺漏 {sizes.loc[CHAOS, 'bed_missing_pct']:.0f}%）、"
       f"直線作答比例 {sizes.loc[CHAOS, 'straight_pct']:.0f}%；日誌與自填作息矛盾，部分可能是作答品質問題，判讀宜保守。"
       if CHAOS in sizes.index else ""),
    "- 國八自傷／求死意念為二元，採加權線性機率模型（係數 ×100 = 百分點）；國七無此題，M2/M3 的前測以國七憂鬱（不含睡眠題）代替。",
    "- M3 加入國七幸福感、生活滿意、成績、私校、受凌／網路受凌、家庭凝聚、父母婚姻；不放補習、網路成癮、FOMO"
    "（補習是類型定義的一部分，成癮與 FOMO 可能是中介），三者放在 M3b 作最保守估計。",
    "- 因果宣稱：類型是觀察性分組。M2/M3 控制國七同一指標後的差異屬「預測性關聯」（預測一年間的相對變化），"
    "仍可能受未測混淆（家庭、學校、既有心理困擾）與回歸均值影響，不應解讀為改變晚上作息即可造成同等改善。",
    "- 敏感度（w2_outcomes_by_type.csv 的 sample 欄）：sens_certain＝排除直線作答與最大後驗 <.8；with_volunteers＝含自願加答者；"
    f"k={K_ALT} 解見 w2_outcomes_by_type_k{K_ALT}.csv；晚睡型合併見 w2_late_types_pooled.csv；深夜 vs 螢幕見 "
    "w2_type_vs_bedtime.csv、w2_late_night_vs_screen.csv、w2_cram_split_by_late_night.csv。",
    f"- 國八追蹤率依類型不同（{(100 * ret).round(0).astype(int).to_dict()}），晚睡／紊亂類型流失較多，"
    "可能使國八差異略為低估。", ""]
md = ["# 國七晚間類型學（scripts/02_evening_typology.py）", "",
      f"執行時間 {time.time() - T0:.0f}s；隨機種子 {SEED}；bootstrap B={N_BOOT}（每次 LCA {N_START_BOOT} 組隨機起始值）；"
      f"LCA 主解隨機起始值 {N_START} 組。", "",
      *summary,
      "## 主要輸出", "",
      "- figures/fig1_evening_types_timeline.png：各類型逐時活動組成＋任何螢幕使用折線（主圖）",
      "- figures/fig2_w2_outcomes_forest.png：類型 → 國八結果（森林圖，M1/M2/M3）",
      "- figures/fig3_type_profiles_w1.png：國七輪廓熱圖",
      "- figures/fig_s1_k_selection.png、fig_s2_sequence_index.png：選 k 與序列索引圖（附錄）",
      "- tables/：k_selection、type_sizes、type_hourly_composition、type_profiles_w1、w2_outcomes_by_type（含敏感度）、"
      f"w2_outcomes_by_type_k{K_ALT}、w2_late_types_pooled、w2_type_vs_bedtime、w2_late_night_vs_screen、"
      "w2_cram_split_by_late_night、w2_descriptives_by_type、method_agreement_ARI、crosstab_LCA_vs_PAM、"
      f"crosstab_k{K_ALT}_vs_k{K_FINAL}、type_assignment、posterior_probs、dhd_mean_substitution_cost", "",
      "## 執行紀錄", "", "```", *LOG, "```"]
(OUT / "results.md").write_text("\n".join(md), encoding="utf-8")
print(f"\n完成（{time.time() - T0:.0f}s）")
