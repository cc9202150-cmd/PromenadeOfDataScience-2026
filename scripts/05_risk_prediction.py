"""國七 → 國八 憂鬱或自傷意念風險早期預警模型（機器學習）。

問題：只用國七（W1）問卷，能否預先找出國八（W2）會出現憂鬱或自傷意念風險的學生？哪些因子相關？
結果定義：國八風險 = CESD≥16 或 近 1–2 週有「想傷害自己／寧願已死」至少一天（門檻寬鬆，新發生個案過半只靠自傷題）。
  (主) 新發生：國七 CESD<16（尚未達篩檢門檻）者，國八轉為風險 —— 輔導室最需要的「提早一年」名單
  (共同主要) 新發生憂鬱症狀：只看 CESD≥16（不含自傷意念）
  (次) 全樣本：國八風險（含國七已高者，持續性主導）
  (敏感度) 新發生、嚴格自傷門檻：CESD≥16 或 任一自傷題 ≥3 天 或 兩題皆 ≥1 天
特徵：只用國七變項（共同題 + A 卷作息/上網/FOMO + B 卷教養）。主卷（共A/共B）以學校為單位分派（89 校 A 卷、90 校 B 卷，同校同卷），
  但約 6–7% 的 A/B 卷作答者是自願加填「附加卷」者（非隨機），另做限主卷的敏感度分析。
  缺答由 HGB 原生處理；邏輯斯迴歸以中位數插補 + 遺漏指標。
評估：以學校為群組的 5 折交叉驗證（StratifiedGroupKFold，超參數於訓練折內再以 3 折群組 CV 選擇），
  所有指標以 HOUWGT 加權；95% CI = 學校層級集群 bootstrap（固定 OOF 預測重抽學校）。
解釋：SHAP（TreeExplainer, HGB 全樣本最終模型；為模型內條件歸因，非因果）、領域組合 SHAP 與「移除領域」ΔAUC；
  晚睡/深夜螢幕/FOMO 以多變項加權邏輯斯迴歸檢驗獨立關聯（共線特徵下的部分依賴圖不可靠，僅作診斷）；
  公平性：依性別、數位發展區，並比較不同門檻政策的取捨。

執行：.venv/Scripts/python -I scripts/05_risk_prediction.py
輸出：outputs/prediction/figures/*.png、outputs/prediction/tables/*.csv、outputs/prediction/results.md
"""
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import partial_dependence
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tigps import data as D  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.titleweight": "bold", "axes.titlesize": 12})
OUT = ROOT / "outputs" / "prediction"
FIG, TAB = OUT / "figures", OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

SEED, NBOOT, TOP, K_SCREEN = 20261019, 1000, .15, 10
COL = {"hgb": "#2a78d6", "screener": "#eb6834", "lr": "#1baf7a", "cesd_only": "#8a8984", "rf": "#4a3aa7"}
MODEL_LAB = {"cesd_only": "僅國七憂鬱分數", "screener": f"{K_SCREEN} 題簡短篩檢", "lr": "邏輯斯迴歸（全特徵）",
             "hgb": "梯度提升樹 HGB（全特徵）", "rf": "隨機森林（全特徵）"}
INK, INK2 = "#0b0b0b", "#52514e"
LOG = []
T0 = time.time()


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def mean_of(df, cols, min_frac=.6):
    return D.scale_mean(df, cols, min_frac)


def flag(cond, src):
    """布林 → 0/1，來源題遺漏則為 NaN。"""
    return cond.astype(float).where(src.notna())


# ======================================================================
# 1. 特徵建構（只用國七）
# ======================================================================
def build_features(raw: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.set_index("nstudent_id")
    r = raw.set_index("nstudent_id").loc[p.index]
    X = pd.DataFrame(index=p.index)
    # --- 國七心理健康 ---
    for v in ["cesd", "who5", "life_sat"]:
        X[v] = p[f"{v}_1"]
    X["happy"], X["health"] = r["as63"], r["as64"]
    X["self_esteem"] = mean_of(r, ["as65a", "as65b", "as65c"])
    # --- 網路使用 ---
    X["addict"], X["fomo"], X["soccomp"] = p["addict_1"], p["fomo_1"], p["soccomp_1"]
    X["prefer_online"] = mean_of(r, ["as38m", "as38n", "as38o"])
    X["escape_cope"] = mean_of(r, ["as37d", "as37e", "as37f"])
    X["support_cope"] = r["as37a"]
    mh = r[["as36_5", "as36_6"]]
    X["search_mh"] = flag((mh == 1).any(axis=1), r["as36_9"].fillna(mh.max(axis=1)))
    nouse = r["as34"] == 2                                  # 平日不上網者跳答 → 0 小時
    for k in D.NET_W1:
        X[f"net_{k}"] = p[f"net_{k}_1"].mask(nouse, 0)
    X["net_leisure"] = X[[f"net_{k}" for k in ("game", "video", "chat", "browse")]].sum(axis=1, min_count=4)
    # 註：as56b 智慧型手機使用頻率分布不合理（A 卷作答者 72.6% = 3,141/4,324 選「幾乎沒有」，疑似選項方向誤植），不納入
    X["study_selfreg"] = mean_of(r, ["as59a", "as59b"])
    X["dig_lit"] = mean_of(r, [f"as61{c}" for c in "abcdefghi"])
    sm_acc = r[[f"as31a{i}" for i in range(1, 11)]].sum(axis=1, min_count=1)
    X["sm_accounts"] = np.where(r["as31"] == 2, 0, sm_acc)
    X["sm_accounts"] = X["sm_accounts"].where(r["as31"].notna())
    # --- 睡眠與夜間作息（A 卷） ---
    for v in ["bedtime", "waketime", "sleep_hrs", "sleep_quality"]:
        X[v] = p[f"{v}_1"]
    X["wkend_bedtime"] = D.clock_bedtime(r["as68c1"], r["as68c2"])
    wk_wake = (r["as68d1"] + r["as68d2"].fillna(0) / 60).where(lambda s: s.between(5, 15))
    X["social_jetlag"] = ((X["wkend_bedtime"] + wk_wake + 24) / 2 - (X["bedtime"] + X["waketime"] + 24) / 2
                          ).where(lambda s: s.between(-4, 8))
    G = pd.DataFrame({h: p[f"diary_{h}_1"].map(D.DIARY_GROUPS) for h in D.DIARY_SLOTS})
    ff = G.ffill(axis=1)
    G = G.fillna(ff.where(ff == "睡覺"))                     # 已睡著後的空白時段 → 睡覺
    screen = G.isin({"螢幕：影音", "螢幕：遊戲/電視", "螢幕：社群聊天"})
    late = ["23", "00", "01"]
    X["late_screen"] = screen[late].any(axis=1).astype(float).where(G[late].notna().any(axis=1))
    X["screen_slots"] = screen.sum(axis=1).astype(float).where(G.notna().sum(axis=1) >= 6)
    X["awake_midnight"] = (G["00"] != "睡覺").astype(float).where(G["00"].notna())
    # --- 學校與同儕 ---
    X["grade_rank"] = p["grade_rank_1"]
    X["keep_up"], X["school_sat"] = r["as19"], r["as13"]
    X["belonging"] = mean_of(pd.DataFrame({"a": r["as14a"], "b": 5 - r["as14b"], "c": r["as14c"]}), ["a", "b", "c"])
    cc = pd.DataFrame({"a": r["as16a"], "b": 5 - r["as16b"], "c": r["as16c"], "d": r["as16d"], "e": r["as16e"]})
    X["class_climate"] = mean_of(cc, list("abcde"))
    X["friend_support"] = mean_of(r, ["as42a", "as42c", "as42d", "as42e"])
    X["deviant_friend"] = r["as42b"]
    X["bully_victim"], X["cyber_victim"] = p["bully_victim_1"], p["cyber_victim_1"]
    for v, q, n in [("bully_sev", "as51", 8), ("cyber_sev", "as53", 7)]:
        sev = r[[f"{q}a{i}" for i in range(1, n + 1)]].mean(axis=1)
        X[v] = pd.Series(np.where(r[q] == 2, 1.0, sev), index=r.index).where(r[q].notna())
    X["bully_perp"], X["cyber_perp"] = flag(r["as52"] == 1, r["as52"]), flag(r["as54"] == 1, r["as54"])
    X["delinq"] = r[[f"as55_{i}" for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13)]].sum(axis=1, min_count=1)
    X["cram"] = p["cram_1"]
    # --- 家庭與教養 ---
    X["fam_econ"], X["parents_married"] = p["fam_econ_1"], p["parents_married_cohab_1"]
    X["live_both"] = flag((r["as8_1"] == 1) & (r["as8_2"] == 1), r["as8_1"])
    X["parent_edu"] = r[["as4d1", "as4d2"]].max(axis=1)
    X["has_sib"] = flag(r["as5"] == 1, r["as5"])
    X["cohesion"] = p["cohesion_1"]
    X["par_acad_req"] = (5 - r["as21"]).where(r["as21"] <= 4)   # 大 = 要求越高
    for v in ["warmth", "hostile", "par_restrict", "par_active", "par_encourage", "par_worry",
              "par_monitor", "par_conflict"]:
        X[v] = p[f"{v}_1"]
    X["harsh"], X["inconsistent"], X["knows_where"] = r["as12i"], r["as12h"], r["as12g"]
    X["par_expect"] = mean_of(r, ["as23a", "as23b", "as23c", "as23d"])
    X["par_involve"] = mean_of(r, [f"as24{c}" for c in "abcdefghij"])
    X["self_manage"] = flag(r["as30"] == 8, r["as30"])
    # as32a 選項順序非單調（4=每週一兩次、5=每月三四次），4↔5 對調成由少到多
    X["im_parent"] = r["as32a"].map({1: 1, 2: 2, 3: 3, 4: 5, 5: 4, 6: 6, 7: 7})
    X["sib_conflict"] = r["as7b"]
    # --- 個人特質與背景 ---
    X["female"] = p["female_1"]
    male, fem = (r["as1"] == 2).to_numpy(), (r["as1"] == 1).to_numpy()   # 青春期發育量表 PDS：共同 3 題 + 性別 2 題
    sexq = lambda c: (r[c] - 1).where(r[c] > 1)             # noqa: E731  選項 1「我是女生/男生」= 不適用 → NaN
    sx = pd.DataFrame({"s1": np.where(male, sexq("as78"), np.where(fem, sexq("as80"), np.nan)),
                       "s2": np.where(male, sexq("as79"), np.where(fem, r["as81"].map({2: 4.0, 3: 1.0}), np.nan))},
                      index=r.index)
    pds = pd.concat([r[["as74", "as75", "as76"]], sx], axis=1)
    X["pds"] = pds.mean(axis=1).where(pds.notna().sum(axis=1) >= 3)
    h, wkg = r["as82"].where(r["as82"].between(130, 195)), r["as83"].where(r["as83"].between(25, 130))
    X["bmi"] = (wkg / (h / 100) ** 2).where(lambda s: s.between(12, 45))
    X["body_size"] = r["as84"]
    X["attach_anx"] = mean_of(r, ["as66d", "as66e", "as66f"])
    X["attach_avoid"] = mean_of(r, ["as66g", "as66h", "as66i"])
    X["self_eff"] = mean_of(r, ["as71d", "as71e", "as71f"])
    X["sugary"], X["handshake"] = r["as69a"], r["as69b"]
    X["digi"] = p["digi_level"].map({"數位發展成熟區": 1, "數位發展潛力區": 2, "數位發展起步區": 3, "數位發展萌動區": 4})
    X["private"] = (p["school_type"] == 2).astype(float)
    X["has_A"], X["has_B"] = p["has_A_1"].astype(float), p["has_B_1"].astype(float)
    return X.astype(float)


FEAT = {  # 名稱: (中文標籤, 領域)
    "cesd": ("國七憂鬱症狀 CESD", "國七心理健康"), "who5": ("國七幸福感 WHO-5", "國七心理健康"),
    "life_sat": ("生活滿意度", "國七心理健康"), "happy": ("最近過得快樂", "國七心理健康"),
    "health": ("自評健康", "國七心理健康"), "self_esteem": ("自尊", "國七心理健康"),
    "addict": ("網路成癮傾向", "網路使用"), "fomo": ("錯失恐懼 FOMO", "網路使用"),
    "soccomp": ("網路向上比較", "網路使用"), "prefer_online": ("偏好網路勝過現實", "網路使用"),
    "escape_cope": ("上網逃避壓力", "網路使用"), "support_cope": ("煩惱時上網找人訴苦", "網路使用"),
    "search_mh": ("曾上網查憂鬱/焦慮", "網路使用"), "net_homework": ("平日上網寫作業時數", "網路使用"),
    "net_learning": ("平日線上學習時數", "網路使用"), "net_game": ("平日遊戲時數", "網路使用"),
    "net_video": ("平日影音時數", "網路使用"), "net_chat": ("平日聊天時數", "網路使用"),
    "net_browse": ("平日瀏覽時數", "網路使用"), "net_leisure": ("平日休閒上網總時數", "網路使用"),
    "study_selfreg": ("讀書時能管住上網", "網路使用"),
    "dig_lit": ("數位素養", "網路使用"), "sm_accounts": ("社群帳號數", "網路使用"),
    "bedtime": ("上課日上床時間", "睡眠與夜間作息"), "waketime": ("上課日起床時間", "睡眠與夜間作息"),
    "sleep_hrs": ("上課日睡眠時數", "睡眠與夜間作息"), "sleep_quality": ("睡眠品質", "睡眠與夜間作息"),
    "wkend_bedtime": ("假日上床時間", "睡眠與夜間作息"), "social_jetlag": ("社會時差", "睡眠與夜間作息"),
    "late_screen": ("23 點後以螢幕為主", "睡眠與夜間作息"), "screen_slots": ("晚間螢幕時段數", "睡眠與夜間作息"),
    "awake_midnight": ("午夜仍醒著", "睡眠與夜間作息"),
    "grade_rank": ("成績排名（大=後段）", "學校與同儕"), "keep_up": ("跟得上課業", "學校與同儕"),
    "school_sat": ("學校生活滿意", "學校與同儕"), "belonging": ("學校歸屬感", "學校與同儕"),
    "class_climate": ("班級氣氛", "學校與同儕"), "friend_support": ("好友支持", "學校與同儕"),
    "deviant_friend": ("好友會惹事", "學校與同儕"), "bully_victim": ("曾遭實體霸凌", "學校與同儕"),
    "bully_sev": ("實體霸凌受害頻率", "學校與同儕"), "cyber_victim": ("曾遭網路霸凌", "學校與同儕"),
    "cyber_sev": ("網路霸凌受害頻率", "學校與同儕"), "bully_perp": ("曾霸凌他人", "學校與同儕"),
    "cyber_perp": ("曾網路霸凌他人", "學校與同儕"), "delinq": ("偏差行為項數", "學校與同儕"),
    "cram": ("有補習", "學校與同儕"),
    "fam_econ": ("家庭經濟（主觀）", "家庭與教養"), "parents_married": ("親生父母婚姻同住", "家庭與教養"),
    "live_both": ("與親生父母同住", "家庭與教養"), "parent_edu": ("照顧者最高學歷", "家庭與教養"),
    "has_sib": ("有兄弟姊妹", "家庭與教養"), "cohesion": ("家庭凝聚力", "家庭與教養"),
    "par_acad_req": ("家長成績要求", "家庭與教養"), "warmth": ("家長溫暖", "家庭與教養"),
    "hostile": ("家長敵意/冷漠", "家庭與教養"), "harsh": ("家長體罰", "家庭與教養"),
    "inconsistent": ("家長管教不一致", "家庭與教養"), "knows_where": ("家長知道行蹤", "家庭與教養"),
    "par_restrict": ("數位教養：限制", "家庭與教養"), "par_active": ("數位教養：陪伴討論", "家庭與教養"),
    "par_encourage": ("數位教養：鼓勵上網", "家庭與教養"), "par_worry": ("數位教養：擔心", "家庭與教養"),
    "par_monitor": ("數位教養：監看", "家庭與教養"), "par_conflict": ("數位教養：責罵衝突", "家庭與教養"),
    "par_expect": ("家長未來期望", "家庭與教養"), "par_involve": ("家長學校參與", "家庭與教養"),
    "self_manage": ("上網由自己管", "家庭與教養"), "im_parent": ("與家長即時通訊頻率", "家庭與教養"),
    "sib_conflict": ("手足衝突", "家庭與教養"),
    "female": ("女生", "個人特質與背景"), "pds": ("青春期發育", "個人特質與背景"),
    "bmi": ("BMI", "個人特質與背景"), "body_size": ("自覺體型（大=胖）", "個人特質與背景"),
    "attach_anx": ("人際焦慮（怕不被喜歡）", "個人特質與背景"), "attach_avoid": ("人際迴避", "個人特質與背景"),
    "self_eff": ("自我效能", "個人特質與背景"), "sugary": ("含糖飲料量", "個人特質與背景"),
    "handshake": ("手搖飲量", "個人特質與背景"), "digi": ("數位發展區域（大=落後）", "個人特質與背景"),
    "private": ("私立學校", "個人特質與背景"), "has_A": ("填 A 卷", "個人特質與背景"),
    "has_B": ("填 B 卷", "個人特質與背景"),
}
FEATS = list(FEAT)

# 簡短篩檢候選題：全體共同作答、單一題目、在教室可直接問
SCREEN_POOL = {
    "as72a": "胃口不好", "as72b": "睡眠情況不好", "as72c": "覺得悲傷", "as72d": "覺得自己是壞人",
    "as72e": "對有興趣的事失去興趣", "as72f": "覺得行動太緩慢", "as72g": "總是感到疲憊",
    "as72h": "無法集中精神", "as72i": "覺得坐立不安", "as72j": "覺得很寂寞", "as72k": "感到緊張焦慮",
    "as72l": "無法停止擔心",
    "as73a": "情緒開朗精神不錯", "as73b": "心情平靜放鬆", "as73c": "有活力精力充沛",
    "as73d": "醒來神清氣爽", "as73e": "生活充滿有趣事物",
    "as62": "對生活滿意", "as63": "最近過得快樂", "as64": "自評健康", "as65a": "我是有價值的人",
    "as65b": "積極樂觀看待自己", "as65c": "滿意自己",
    "as41a": "無法上網就坐立不安", "as41b": "沒上網心裡不舒服", "as41c": "超過一天沒上網很難受",
    "as41d": "上網時間越來越長", "as41e": "上網比以前多很多", "as41f": "上網常超過預期",
    "as41g": "上網影響學業", "as41h": "上網危害身心健康", "as41i": "上網減少與家人朋友互動",
    "as41j": "熬夜上網白天沒精神",
    "as9a": "家人做決定會商量", "as9b": "家人喜歡共度休閒", "as9c": "家庭活動大家都參加",
    "as9d": "家人接納彼此朋友", "as9e": "受挫時家人會安慰", "as9f": "需要時可依賴家人",
    "as13": "學校生活滿意", "as14a": "喜歡學校", "as14b": "不覺得是學校一份子", "as14c": "以學校為榮",
    "as16a": "同學會適時伸出援手", "as16b": "不喜歡與同學來往", "as16c": "我會幫助同學",
    "as16d": "班上相親相愛", "as16e": "班上同學互相信賴",
    "as42a": "可依賴好友", "as42b": "好友會惹事", "as42c": "好友關心我", "as42d": "常得到好友幫忙",
    "as42e": "受挫時好友會安慰", "as19": "跟得上課業", "as20": "成績排名（大=後段）",
    "as10": "家庭經濟", "is_female": "女生", "is_bullied": "曾遭實體霸凌", "is_cyber": "曾遭網路霸凌",
    "as59a": "讀書時關掉干擾", "as59b": "先完成作業才玩",
}


def build_items(raw: pd.DataFrame, ids: pd.Index) -> pd.DataFrame:
    r = raw.set_index("nstudent_id").loc[ids]
    it = r[[k for k in SCREEN_POOL if k.startswith("as")]].copy()
    it["is_female"] = flag(r["as1"] == 1, r["as1"])
    it["is_bullied"] = flag(r["as51"] == 1, r["as51"])
    it["is_cyber"] = flag(r["as53"] == 1, r["as53"])
    return it.astype(float)


# ======================================================================
# 2. 評估工具（加權）
# ======================================================================
def wauc(y, s, w):
    """加權 ROC-AUC（同分以 0.5 計）。"""
    _, idx = np.unique(s, return_inverse=True)
    wp, wn = np.bincount(idx, w * y), np.bincount(idx, w * (1 - y))
    below = np.cumsum(wn) - wn
    den = wp.sum() * wn.sum()
    return np.sum(wp * (below + .5 * wn)) / den if den > 0 else np.nan


def wap(y, s, w):
    return average_precision_score(y, s, sample_weight=w) if y.any() else np.nan


def wbrier(y, s, w):
    return np.average((s - y) ** 2, weights=w)


def wquantile(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return x[o][min(np.searchsorted(cw, q), len(x) - 1)]


def screen_stats(y, s, w, frac=TOP, thr=None):
    """標記前 frac（加權）高風險者：敏感度、PPV、特異度、NPV、實際標記比例。"""
    thr = wquantile(s, w, 1 - frac) if thr is None else thr
    f = s >= thr
    wy, wf = w * y, w * f
    sens = (wf * y).sum() / wy.sum() if wy.sum() > 0 else np.nan
    ppv = (wf * y).sum() / wf.sum() if wf.sum() > 0 else np.nan
    spec = (w * (~f) * (1 - y)).sum() / (w * (1 - y)).sum()
    npv = (w * (~f) * (1 - y)).sum() / (w * (~f)).sum()
    return sens, ppv, spec, npv, wf.sum() / w.sum()


def cal_slope(y, s, w):
    lp = np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6)))
    slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial(), var_weights=w).fit().params[1]
    citl = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp, var_weights=w).fit().params[0]
    return slope, citl


def cluster_boot(fn, g, B=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq = np.unique(g)
    idx = [np.flatnonzero(g == s) for s in uniq]
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        out.append(fn(np.concatenate([idx[k] for k in pick])))
    return np.array(out, dtype=float)


def ci_str(est, lo, hi, nd=3):
    return f"{est:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def pfmt(p):
    return "p<.001" if p < .001 else f"p={p:.3f}"


# ======================================================================
# 3. 模型
# ======================================================================
LR_GRID = [1e-4, 3e-4, .001, .003, .01, .03, .1, 1.0]
HGB_GRID = [dict(learning_rate=.05, max_iter=150, max_leaf_nodes=8, min_samples_leaf=40),
            dict(learning_rate=.05, max_iter=300, max_leaf_nodes=8, min_samples_leaf=40),
            dict(learning_rate=.03, max_iter=300, max_depth=3, min_samples_leaf=80),
            dict(learning_rate=.03, max_iter=600, max_depth=2, min_samples_leaf=80)]
RF_PRM = dict(n_estimators=500, min_samples_leaf=25, max_features=.3)


def fit(kind, prm, X, y, w):
    if kind == "lr":
        m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                          LogisticRegression(C=prm, max_iter=5000))
        return m.fit(X, y, logisticregression__sample_weight=w)
    if kind == "hgb":
        m = HistGradientBoostingClassifier(**prm, l2_regularization=1.0, max_features=.5,
                                           early_stopping=False, random_state=SEED)
    else:
        m = RandomForestClassifier(**prm, n_jobs=-1, random_state=SEED)
    return m.fit(X, y, sample_weight=w)


def tune(kind, grid, X, y, w, g):
    if len(grid) == 1:
        return grid[0]
    cv = list(StratifiedGroupKFold(3, shuffle=True, random_state=SEED).split(X, y, g))
    sc = [np.mean([wauc(y[te], fit(kind, prm, X.iloc[tr], y[tr], w[tr]).predict_proba(X.iloc[te])[:, 1], w[te])
                   for tr, te in cv]) for prm in grid]
    return grid[int(np.argmax(sc))]


def cross_predict(kind, grid, X, y, w, g, folds):
    oof, chosen, models = np.zeros(len(y)), [], []
    for tr, te in folds:
        prm = tune(kind, grid, X.iloc[tr], y[tr], w[tr], g[tr])
        m = fit(kind, prm, X.iloc[tr], y[tr], w[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        chosen.append(prm)
        models.append(m)
    return oof, chosen, models


def prep(Xtr, Xte):
    """訓練折中位數插補 + 標準化（簡短篩檢用）。"""
    med = np.nanmedian(Xtr, axis=0)
    Ztr, Zte = np.where(np.isnan(Xtr), med, Xtr), np.where(np.isnan(Xte), med, Xte)
    mu, sd = Ztr.mean(0), Ztr.std(0)
    sd[sd == 0] = 1
    return (Ztr - mu) / sd, (Zte - mu) / sd


def forward_select(Z, y, w, k=K_SCREEN):
    """逐步前進選題：每步加入使（訓練資料）加權 AUC 增加最多的一題。"""
    sel, rest = [], list(range(Z.shape[1]))
    for _ in range(k):
        best = (-1, None)
        for j in rest:
            cols = sel + [j]
            m = LogisticRegression(C=1.0, max_iter=2000).fit(Z[:, cols], y, sample_weight=w)
            a = wauc(y, Z[:, cols] @ m.coef_[0], w)
            if a > best[0]:
                best = (a, j)
        sel.append(best[1])
        rest.remove(best[1])
    return sel


def run_screener(I, y, w, folds):
    """巢狀：每一外折內自行選題（避免選題洩漏），回傳 1..K 題的 OOF 預測與各折選題。"""
    Iv = I.to_numpy()
    oof = np.zeros((len(y), K_SCREEN))
    fold_sel = []
    for tr, te in folds:
        Ztr, Zte = prep(Iv[tr], Iv[te])
        sel = forward_select(Ztr, y[tr], w[tr])
        fold_sel.append([I.columns[j] for j in sel])
        for k in range(1, K_SCREEN + 1):
            m = LogisticRegression(C=1.0, max_iter=2000).fit(Ztr[:, sel[:k]], y[tr], sample_weight=w[tr])
            oof[te, k - 1] = m.predict_proba(Zte[:, sel[:k]])[:, 1]
    Zall, _ = prep(Iv, Iv)
    final = [I.columns[j] for j in forward_select(Zall, y, w)]
    return oof, fold_sel, final


# ======================================================================
# 4. 資料與結果變項
# ======================================================================
log("# 國七 → 國八 憂鬱或自傷意念風險早期預警模型\n")
panel = D.load_panel()
raw1, _ = D.read_wave(D.W1_SAV)
raw2, _ = D.read_wave(D.W2_SAV)                              # 只用於結果變項（自傷題原始頻率）
F = build_features(raw1, panel)
assert not any(c.endswith("_2") for c in F.columns), "特徵不可含國八變項"
I = build_items(raw1, F.index)
p = panel.set_index("nstudent_id")
df = F.join(I).join(p[["nschool_id", "digi_level", "HOUWGT", "cesd_2", "selfharm_2", "cesd_nosleep_1",
                       "cesd_nosleep_2"]]).copy()
df["resp"] = raw1.set_index("nstudent_id").loc[df.index, "response"]
df["main_A"] = df.resp.isin([1, 3]).astype(float)             # 主卷 = A（以學校為單位分派）；附加卷為自願加填
df["main_B"] = df.resp.isin([2, 4]).astype(float)
sh = raw2.set_index("nstudent_id").reindex(df.index)[D.SELFHARM_W2]
sh_strict = ((sh.max(axis=1) >= 2) | ((sh >= 1).sum(axis=1) >= 2)).astype(float).where(sh.notna().any(axis=1))
df["sh_max"] = sh.max(axis=1)
df["sh_n1"] = (sh >= 1).sum(axis=1)
hr = np.where((df.cesd_2 >= 16) | (df.selfharm_2 == 1), 1.0,
              np.where(df.cesd_2.notna() & df.selfharm_2.notna(), 0.0, np.nan))
df["y_full"] = hr
df["y_inc"] = df["y_full"].where(df["cesd"] < 16)
df["y_inc_cesd"] = (df["cesd_2"] >= 16).astype(float).where(df["cesd_2"].notna() & (df["cesd"] < 16))
df["y_inc_strict"] = pd.Series(np.where((df.cesd_2 >= 16) | (sh_strict == 1), 1.0,
                                        np.where(df.cesd_2.notna() & sh_strict.notna(), 0.0, np.nan)),
                               index=df.index).where(df["cesd"] < 16)
# 無睡眠題版本（CESD 去掉睡眠題，11 題 ×11，門檻等比例 16×11/12），供睡眠相關推論的敏感度分析
df["y_inc_ns"] = pd.Series(np.where((df.cesd_nosleep_2 >= 16 * 11 / 12) | (df.selfharm_2 == 1), 1.0,
                                    np.where(df.cesd_nosleep_2.notna() & df.selfharm_2.notna(), 0.0, np.nan)),
                           index=df.index).where(df["cesd"] < 16)
df = df.copy()
TARGETS = {"y_inc": "新發生憂鬱或自傷意念風險（國七 CESD<16 → 國八 CESD≥16 或近兩週自傷/求死意念）",
           "y_inc_cesd": "新發生憂鬱症狀（國七 CESD<16 → 國八 CESD≥16；不含自傷意念）",
           "y_full": "國八憂鬱或自傷意念風險（全樣本）",
           "y_inc_strict": "新發生（嚴格自傷門檻：CESD≥16 或任一自傷題≥3 天或兩題皆有）"}
TSHORT = {"y_inc": "新發生憂鬱或自傷意念", "y_inc_cesd": "新發生憂鬱症狀（僅 CESD）", "y_full": "全樣本國八風險",
          "y_inc_strict": "新發生（嚴格自傷門檻）"}

log("## 樣本與盛行率（加權 HOUWGT；CI = 學校集群 bootstrap）")
hw = df.HOUWGT.notna()
yok = hw & df.y_full.notna()
flow = [("兩波串接（nstudent_id）", len(df)), ("有權重 HOUWGT", int(hw.sum())),
        ("國八結果可判定（全樣本目標）", int(yok.sum())),
        ("　排除：國七 CESD 遺漏", int((yok & df.cesd.isna()).sum())),
        ("　排除：國七 CESD≥16（已達門檻）", int((yok & (df.cesd >= 16)).sum())),
        ("新發生世代（國七 CESD<16）", int((yok & (df.cesd < 16)).sum()))]
pd.DataFrame(flow, columns=["步驟", "n"]).to_csv(TAB / "sample_flow.csv", index=False, encoding="utf-8-sig")
log("樣本流程：" + " → ".join(f"{a.strip('　')} {b:,}" for a, b in flow) +
    f"；學校數 {df.nschool_id.nunique()}。（另有 {int((hw & df.y_full.isna() & (df.cesd < 16)).sum())} 位國七 CESD<16 者"
    "國八結果無法判定，未納入）")
prev_rows = []
for t, lab in TARGETS.items():
    d = df[df[t].notna() & df.HOUWGT.notna()]
    y, w, g = d[t].to_numpy(), d.HOUWGT.to_numpy(), pd.factorize(d.nschool_id)[0]
    est = np.average(y, weights=w)
    bs = cluster_boot(lambda ii: [np.average(y[ii], weights=w[ii])], g, 500)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    prev_rows.append(dict(結果=lab, n=len(d), 事件數=int(y.sum()), 加權盛行率=est, CI下=lo, CI上=hi))
    log(f"- {lab}: n={len(d)}, 事件 {int(y.sum())}, 加權盛行率 {est:.1%} [{lo:.1%}, {hi:.1%}]")
pd.DataFrame(prev_rows).round(4).to_csv(TAB / "target_prevalence.csv", index=False, encoding="utf-8-sig")
d0 = df[df.y_full.notna() & hw]
log(f"  （全樣本國八：CESD≥16 {np.average(d0.cesd_2 >= 16, weights=d0.HOUWGT):.1%}，"
    f"自傷/求死意念 {np.average(d0.selfharm_2, weights=d0.HOUWGT):.1%}）")
c0 = df[(df.y_inc == 1) & hw]
sh_only = (c0.cesd_2 < 16).to_numpy()
weak = sh_only & (c0.sh_max == 1).to_numpy() & (c0.sh_n1 == 1).to_numpy()
COMP = dict(n_case=len(c0), n_sh_only=int(sh_only.sum()), w_sh_only=np.average(sh_only, weights=c0.HOUWGT),
            n_weak=int(weak.sum()), w_weak=np.average(weak, weights=c0.HOUWGT))
log(f"  新發生個案組成：{COMP['n_case']:,} 人中 {COMP['n_sh_only']:,} 人（加權 {COMP['w_sh_only']:.1%}）國八 CESD<16、"
    f"只因自傷/求死意念題入列；其中 {COMP['n_weak']:,} 人（加權 {COMP['w_weak']:.1%}）只有單一題、最低頻率（最近 1 週 1–2 天）。"
    "→ 以「憂鬱或自傷意念風險」稱之，並把只看 CESD 的目標列為共同主要結果。")

# 題本歸屬（以實際作答率判定）與遺漏率
book = {}
for f in FEATS:
    oa = df.loc[df.has_A == 1, f].notna().mean()
    ob = df.loc[df.has_B == 1, f].notna().mean()
    only_a = df.loc[(df.has_A == 1) & (df.has_B == 0), f].notna().mean()
    only_b = df.loc[(df.has_A == 0) & (df.has_B == 1), f].notna().mean()
    book[f] = "A" if only_b < .05 and oa > .5 else ("B" if only_a < .05 and ob > .5 else "共同")
feat_tab = pd.DataFrame({"特徵": FEATS, "中文": [FEAT[f][0] for f in FEATS], "領域": [FEAT[f][1] for f in FEATS],
                         "題本": [book[f] for f in FEATS],
                         "遺漏率": [df[f].isna().mean() for f in FEATS],
                         "平均": [df[f].mean() for f in FEATS]}).round(3)
feat_tab.to_csv(TAB / "features.csv", index=False, encoding="utf-8-sig")
log(f"\n特徵數 {len(FEATS)}（共同 {sum(v == '共同' for v in book.values())}、A 卷 {sum(v == 'A' for v in book.values())}、"
    f"B 卷 {sum(v == 'B' for v in book.values())}）；簡短篩檢候選題 {len(SCREEN_POOL)} 題（皆為全體共同題）")
di = df[df.y_inc.notna() & hw]
volA, volB = di.resp.isin([2, 5]), di.resp == 1
BOOKV = dict(nA=int((di.has_A == 1).sum()), volA=int(volA.sum()), nB=int((di.has_B == 1).sum()), volB=int(volB.sum()),
             fem_vol1=(di.loc[di.resp == 1, "female"]).mean(), fem_main=di.loc[di.resp.isin([3, 4]), "female"].mean())
log(f"題本：主卷（共A/共B）以學校為單位分派（同校同卷）；另有自願加填的附加卷。新發生世代中 A 卷作答者 {BOOKV['nA']:,} 人含附加 A 卷自願者 "
    f"{BOOKV['volA']}（{BOOKV['volA'] / BOOKV['nA']:.1%}），B 卷作答者 {BOOKV['nB']:,} 人含附加 B 卷自願者 {BOOKV['volB']}"
    f"（{BOOKV['volB'] / BOOKV['nB']:.1%}）；「共A附B」女生比例 {BOOKV['fem_vol1']:.0%} vs 只填主卷者 {BOOKV['fem_main']:.0%}"
    " → A/B 子樣本為兩組學校而非個人隨機，另以限主卷作答者做敏感度分析（第 7 節）。")


def label(f):
    tag = {"A": "（A卷）", "B": "（B卷）"}.get(book.get(f, ""), "")
    return FEAT[f][0] + tag


# ======================================================================
# 5. 交叉驗證：各目標 × 各模型
# ======================================================================
def run_target(t, screener=True):
    d = df[df[t].notna() & df.HOUWGT.notna()]
    X, y = d[FEATS], d[t].to_numpy(int)
    w, g = d.HOUWGT.to_numpy(float), pd.factorize(d.nschool_id)[0]
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, g))
    R = dict(d=d, X=X, y=y, w=w, g=g, folds=folds, pred={}, chosen={})
    R["pred"]["cesd_only"], _, _ = cross_predict("lr", [1.0], X[["cesd"]], y, w, g, folds)
    R["pred"]["lr"], R["chosen"]["lr"], _ = cross_predict("lr", LR_GRID, X, y, w, g, folds)
    R["pred"]["hgb"], R["chosen"]["hgb"], R["hgb_models"] = cross_predict("hgb", HGB_GRID, X, y, w, g, folds)
    R["pred"]["rf"], _, _ = cross_predict("rf", [RF_PRM], X, y, w, g, folds)
    if screener:
        R["scr_oof"], R["scr_folds"], R["scr_final"] = run_screener(d[list(SCREEN_POOL)], y, w, folds)
        R["pred"]["screener"] = R["scr_oof"][:, -1]
    log(f"  [{t}] CV 完成（{time.time() - T0:.0f}s）")
    return R


def perf(R, t):
    y, w, g = R["y"], R["w"], R["g"]
    models = list(R["pred"])
    P = np.column_stack([R["pred"][m] for m in models])

    def stats(ii):
        out = []
        for j in range(P.shape[1]):
            s = screen_stats(y[ii], P[ii, j], w[ii])
            out += [wauc(y[ii], P[ii, j], w[ii]), wap(y[ii], P[ii, j], w[ii]), wbrier(y[ii], P[ii, j], w[ii]), s[0], s[1]]
        return out

    pt = np.array(stats(np.arange(len(y))))
    bs = cluster_boot(stats, g)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    prev = np.average(y, weights=w)
    rows = []
    for j, m in enumerate(models):
        k = slice(5 * j, 5 * j + 5)
        slope, citl = cal_slope(y, P[:, j], w)
        rows.append(dict(目標=t, 模型=MODEL_LAB[m], key=m, n=len(y), 盛行率=prev,
                         **{f"{n}": v for n, v in zip(["AUC", "PR_AUC", "Brier", f"敏感度@前{TOP:.0%}", f"PPV@前{TOP:.0%}"], pt[k])},
                         **{f"{n}_lo": v for n, v in zip(["AUC", "PR_AUC", "Brier", "敏感度", "PPV"], lo[k])},
                         **{f"{n}_hi": v for n, v in zip(["AUC", "PR_AUC", "Brier", "敏感度", "PPV"], hi[k])},
                         校準斜率=slope, 校準截距=citl))
    tab = pd.DataFrame(rows)
    # 配對 ΔAUC（同一 bootstrap 樣本）
    ia = {m: 5 * j for j, m in enumerate(models)}
    deltas = []
    for a, b in [("hgb", "cesd_only"), ("lr", "cesd_only"), ("hgb", "lr"), ("rf", "hgb"),
                 ("screener", "cesd_only"), ("hgb", "screener")]:
        if a in ia and b in ia:
            dd = bs[:, ia[a]] - bs[:, ia[b]]
            est = pt[ia[a]] - pt[ia[b]]
            pval = min(1, 2 * min((dd <= 0).mean(), (dd >= 0).mean()))
            deltas.append(dict(目標=t, 比較=f"{MODEL_LAB[a]} − {MODEL_LAB[b]}", ΔAUC=est,
                               CI下=np.percentile(dd, 2.5), CI上=np.percentile(dd, 97.5), p_boot=pval))
    return tab, pd.DataFrame(deltas)


log("\n## 模型表現（5 折學校群組 CV，OOF 預測；加權）")
RES = {}
for t, scr in [("y_inc", True), ("y_inc_cesd", True), ("y_full", True), ("y_inc_strict", False)]:
    RES[t] = run_target(t, scr)
perf_tabs, delta_tabs = zip(*[perf(RES[t], t) for t in RES])
perf_tab, delta_tab = pd.concat(perf_tabs), pd.concat(delta_tabs)
perf_tab.round(4).to_csv(TAB / "model_performance.csv", index=False, encoding="utf-8-sig")
delta_tab.round(4).to_csv(TAB / "auc_differences.csv", index=False, encoding="utf-8-sig")
for t in RES:
    sub = perf_tab[perf_tab.目標 == t]
    log(f"\n### {TARGETS[t]}（n={sub.n.iloc[0]}，加權盛行率 {sub.盛行率.iloc[0]:.1%}）")
    log("| 模型 | ROC-AUC | PR-AUC | Brier | 敏感度@前15% | PPV@前15% | 校準斜率 |")
    log("|---|---|---|---|---|---|---|")
    for _, r in sub.iterrows():
        log(f"| {r.模型} | {ci_str(r.AUC, r.AUC_lo, r.AUC_hi)} | {ci_str(r.PR_AUC, r.PR_AUC_lo, r.PR_AUC_hi)} | "
            f"{r.Brier:.3f} | {r[f'敏感度@前{TOP:.0%}']:.1%} [{r.敏感度_lo:.1%}, {r.敏感度_hi:.1%}] | "
            f"{r[f'PPV@前{TOP:.0%}']:.1%} [{r.PPV_lo:.1%}, {r.PPV_hi:.1%}] | {r.校準斜率:.2f} |")
    for _, r in delta_tab[delta_tab.目標 == t].iterrows():
        log(f"- ΔAUC {r.比較}: {r.ΔAUC:+.3f} [{r.CI下:+.3f}, {r.CI上:+.3f}], p_boot={r.p_boot:.3f}")
hp = pd.DataFrame([dict(目標=t, 折=i + 1, LR_C=RES[t]["chosen"]["lr"][i], HGB=str(RES[t]["chosen"]["hgb"][i]))
                   for t in RES for i in range(5)])
hp.to_csv(TAB / "chosen_hyperparams.csv", index=False, encoding="utf-8-sig")

# 不同篩檢比例
PRI = RES["y_inc"]
thr_rows = []
for m in ["hgb", "screener", "lr", "cesd_only"]:
    for frac in [.05, .10, .15, .20, .30]:
        s = screen_stats(PRI["y"], PRI["pred"][m], PRI["w"], frac)
        prev = np.average(PRI["y"], weights=PRI["w"])
        thr_rows.append(dict(模型=MODEL_LAB[m], 標記比例=frac, 敏感度=s[0], PPV=s[1], 特異度=s[2], NPV=s[3],
                             提升倍數=s[1] / prev))
thr_tab = pd.DataFrame(thr_rows).round(3)
thr_tab.to_csv(TAB / "screening_thresholds_incident.csv", index=False, encoding="utf-8-sig")
log("\n### 新發生目標：不同篩檢比例（點估計）")
log(thr_tab[thr_tab.模型.isin([MODEL_LAB["hgb"], MODEL_LAB["screener"]])].to_string(index=False))

# ======================================================================
# 6. 簡短篩檢（≤10 題）
# ======================================================================
log("\n## 簡短篩檢（每外折內逐步選題，避免選題洩漏）")
scr_rows, curve_rows = [], []
for t in ["y_inc", "y_inc_cesd", "y_full"]:
    R = RES[t]
    y, w, g = R["y"], R["w"], R["g"]
    stats = lambda ii: [wauc(y[ii], R["scr_oof"][ii, k], w[ii]) for k in range(K_SCREEN)]  # noqa: E731
    pt, bs = np.array(stats(np.arange(len(y)))), cluster_boot(stats, g)   # 與模型表現同一組 bootstrap 抽樣
    for k in range(K_SCREEN):
        curve_rows.append(dict(目標=t, 題數=k + 1, AUC=pt[k], CI下=np.percentile(bs[:, k], 2.5),
                               CI上=np.percentile(bs[:, k], 97.5)))
    # 全樣本最終題目：OR / 每 1 分與每 1 SD，及各折入選次數
    items = R["scr_final"]
    Xi = R["d"][items]
    med = Xi.median()
    Z = Xi.fillna(med)
    zs = (Z - Z.mean()) / Z.std(ddof=0)
    m = LogisticRegression(C=1.0, max_iter=2000).fit(zs, y, sample_weight=w)
    for rank, (it, b) in enumerate(zip(items, m.coef_[0]), 1):
        scr_rows.append(dict(目標=t, 入選順序=rank, 題號=it, 題目=SCREEN_POOL[it], OR_每SD=np.exp(b),
                             OR_每1分=np.exp(b / Z[it].std(ddof=0)),
                             各折入選次數=sum(it in s for s in R["scr_folds"])))
    log(f"- [{t}] 10 題 CV AUC = {ci_str(pt[-1], np.percentile(bs[:, -1], 2.5), np.percentile(bs[:, -1], 97.5))}；"
        f"5 題 {pt[4]:.3f}、3 題 {pt[2]:.3f}、1 題 {pt[0]:.3f}")
    log("  最終題目（全樣本選題順序）：" + "、".join(f"{SCREEN_POOL[i]}" for i in items))
scr_tab, curve_tab = pd.DataFrame(scr_rows).round(3), pd.DataFrame(curve_rows).round(4)
scr_tab.to_csv(TAB / "screener_items.csv", index=False, encoding="utf-8-sig")
curve_tab.to_csv(TAB / "screener_auc_by_k.csv", index=False, encoding="utf-8-sig")

# 加問 A 卷（作息/FOMO）或 B 卷（教養）能否讓 10 題篩檢更準？（在各自子樣本中、同折內比較）
log("\n## 增益分析：10 題篩檢 + 額外題組（新發生目標；只在有作答該題本者中比較）")
EXTRA = {"A": ("A", ["bedtime", "fomo", "late_screen"], "上床時間 + FOMO + 23 點後螢幕"),
         "A_night": ("A", ["bedtime", "late_screen", "awake_midnight"], "上床時間 + 23 點後螢幕 + 午夜仍醒著"),
         "A_fomo": ("A", ["fomo"], "FOMO"),
         "B": ("B", ["warmth", "hostile", "par_restrict", "par_active", "par_monitor", "par_conflict"],
               "家長溫暖/敵意 + 4 種數位教養")}
inc_rows = []
for key, (bk, extra, lab) in EXTRA.items():
    R = PRI
    d = R["d"]
    sub = (d[f"has_{bk}"] == 1).to_numpy()
    p0, p1 = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for (tr, te), items in zip(R["folds"], R["scr_folds"]):
        tr, te = tr[sub[tr]], te[sub[te]]
        for cols, out in [(items, p0), (items + extra, p1)]:
            Ztr, Zte = prep(d[cols].to_numpy()[tr], d[cols].to_numpy()[te])
            mm = LogisticRegression(C=1.0, max_iter=2000).fit(Ztr, R["y"][tr], sample_weight=R["w"][tr])
            out[te] = mm.predict_proba(Zte)[:, 1]
    ii0 = np.flatnonzero(sub)
    y, w, g = R["y"][ii0], R["w"][ii0], R["g"][ii0]
    q0, q1 = p0[ii0], p1[ii0]
    st = lambda ii: [wauc(y[ii], q0[ii], w[ii]), wauc(y[ii], q1[ii], w[ii])]  # noqa: E731
    pt, bs = np.array(st(np.arange(len(y)))), cluster_boot(st, g)
    dd = bs[:, 1] - bs[:, 0]
    pval = min(1, 2 * min((dd <= 0).mean(), (dd >= 0).mean()))
    inc_rows.append(dict(key=key, 題本=bk, 加入=lab, n=len(y), AUC_10題=pt[0], AUC_加入後=pt[1], ΔAUC=pt[1] - pt[0],
                         CI下=np.percentile(dd, 2.5), CI上=np.percentile(dd, 97.5), p_boot=pval))
    log(f"- {bk} 卷子樣本 n={len(y)}：10 題 AUC {pt[0]:.3f} → 加入「{lab}」{pt[1]:.3f}，"
        f"ΔAUC {pt[1] - pt[0]:+.3f} [{np.percentile(dd, 2.5):+.3f}, {np.percentile(dd, 97.5):+.3f}], p_boot={pval:.3f}")
pd.DataFrame(inc_rows).round(4).to_csv(TAB / "incremental_value.csv", index=False, encoding="utf-8-sig")
_gap = delta_tab.set_index(["目標", "比較"]).loc[("y_inc", f"{MODEL_LAB['hgb']} − {MODEL_LAB['screener']}"), "ΔAUC"]
log(f"  （對照：全特徵 HGB 與 10 題篩檢的 AUC 差距為 {_gap:.3f}。ΔAUC 小但 CI 不含 0 者，代表確實帶來額外的區辨資訊。）")

# ======================================================================
# 7. 推論：控制國七憂鬱與性別後，各國七因子與新發生風險的關聯（BH-FDR）
# ======================================================================
log("\n## 關聯推論：加權邏輯斯迴歸（GEE 獨立相關 = 加權 GLM + 學校集群穩健 SE），控制國七 CESD + 性別，BH-FDR（新發生目標）")
d = PRI["d"].assign(y=PRI["y"], g=PRI["g"])


def gee_or(data, f, cov, yname="y"):
    """加權邏輯斯 GEE（獨立相關 = 加權 GLM + 學校集群穩健 SE）；連續變項以 1 SD 為單位（SD 取自分析樣本）。"""
    dd = data[[f, yname, "HOUWGT", "g"] + cov].dropna()
    binary = dd[f].nunique() <= 2
    dd = dd.assign(z=dd[f] if binary else (dd[f] - dd[f].mean()) / dd[f].std())
    fit_ = smf.gee(f"{yname} ~ z + {' + '.join(cov)}", groups="g", data=dd, family=sm.families.Binomial(),
                   cov_struct=sm.cov_struct.Independence(), weights=dd.HOUWGT).fit()
    b, se = fit_.params["z"], fit_.bse["z"]
    return dict(單位="1 vs 0" if binary else "每 1 SD", n=len(dd), OR=np.exp(b), CI下=np.exp(b - 1.96 * se),
                CI上=np.exp(b + 1.96 * se), p=fit_.pvalues["z"])


or_rows = []
for f in [f for f in FEATS if f not in ("has_A", "has_B")]:
    or_rows.append(dict(特徵=f, 中文=FEAT[f][0], 領域=FEAT[f][1], 題本=book[f],
                        **gee_or(d, f, [c for c in ("cesd", "female") if c != f])))
or_tab = pd.DataFrame(or_rows)
or_tab["q_FDR"] = multipletests(or_tab.p, method="fdr_bh")[1]
or_tab = or_tab.sort_values("p")
or_tab.round(4).to_csv(TAB / "adjusted_or_fdr.csv", index=False, encoding="utf-8-sig")
log(f"{(or_tab.q_FDR < .05).sum()} / {len(or_tab)} 個因子 q<.05。重點因子：")
for f in ["bedtime", "sleep_hrs", "late_screen", "awake_midnight", "fomo", "addict", "net_leisure", "soccomp",
          "warmth", "hostile", "cohesion", "par_restrict", "par_active", "par_monitor", "par_conflict",
          "cyber_victim", "bully_victim", "female", "who5"]:
    r = or_tab[or_tab.特徵 == f].iloc[0]
    log(f"- {r.中文}（{r.單位}, n={r.n}）: OR {r.OR:.2f} [{r.CI下:.2f}, {r.CI上:.2f}], p={r.p:.4f}, q={r.q_FDR:.4f}")

# 7b. 晚睡／深夜螢幕／FOMO 是「獨立相關」還是被其他風險吸收？＋ 題目重疊與題本自選的敏感度分析
log("\n## 敏感度與多變項推論（新發生目標；加權 GEE、學校集群 SE）")
MV_COV = ["cesd", "female", "addict", "sleep_quality", "self_esteem", "life_sat", "who5", "prefer_online"]
SENS_MODELS = {
    "M1": ("控制國七 CESD + 性別（主分析）", lambda f: [c for c in ("cesd", "female") if c != f], "y", None),
    "M2": ("多變項：+ 網路成癮、睡眠品質、自尊、生活滿意、WHO-5、偏好網路", lambda f: [c for c in MV_COV if c != f], "y", None),
    "M3": ("無睡眠題：結果與控制皆用去睡眠題 CESD", lambda f: ["cesd_nosleep_1", "female"], "y_inc_ns", None),
    "M4": ("限主卷作答者（排除自願附加卷）", lambda f: [c for c in ("cesd", "female") if c != f], "y", "main"),
}
NIGHT = ["bedtime", "awake_midnight", "late_screen", "screen_slots", "sleep_hrs", "fomo", "soccomp", "net_leisure",
         "addict"]
PARENT = ["warmth", "hostile", "par_restrict", "par_active", "par_monitor", "par_conflict"]
sens_rows = []
for mk, (mlab, covf, yname, restrict) in SENS_MODELS.items():
    for f in NIGHT + PARENT:
        if (mk in ("M2", "M3") and f in PARENT) or (mk == "M4" and book[f] == "共同"):
            continue
        dat = d if restrict is None else d[d[f"main_{book[f]}"] == 1]
        sens_rows.append(dict(模型=mk, 模型說明=mlab, 特徵=f, 中文=FEAT[f][0], 題本=book[f],
                              共變項="、".join(covf(f)), **gee_or(dat, f, covf(f), yname)))
sens_tab = pd.DataFrame(sens_rows)
sens_tab["q_FDR_模型內"] = sens_tab.groupby("模型").p.transform(lambda s: multipletests(s, method="fdr_bh")[1])
sens_tab.round(4).to_csv(TAB / "adjusted_or_sensitivity.csv", index=False, encoding="utf-8-sig")
SENS = sens_tab.set_index(["模型", "特徵"])
for mk, (mlab, *_) in SENS_MODELS.items():
    sub = sens_tab[sens_tab.模型 == mk]
    log(f"- {mk} {mlab}：" + "；".join(f"{r.中文} {r.OR:.2f} [{r.CI下:.2f}, {r.CI上:.2f}] {pfmt(r.p)} (n={r.n})"
                                     for r in sub.itertuples()))
dA = d[d.has_A == 1]
CORR = {"bedtime~sleep_hrs": dA.bedtime.corr(dA.sleep_hrs), "bedtime~wkend_bedtime": dA.bedtime.corr(dA.wkend_bedtime),
        "late_screen~awake_midnight": dA.late_screen.corr(dA.awake_midnight),
        "bedtime~late_screen": dA.bedtime.corr(dA.late_screen), "bedtime~addict": dA.bedtime.corr(dA.addict)}
log("  A 卷作答者特徵相關（Pearson r）：" + "、".join(f"{k} {v:.2f}" for k, v in CORR.items()) +
    "。上床時間與睡眠時數幾乎是同一資訊（睡眠時數由上床/起床時間算出），網路成癮量表本身含「熬夜上網」題（as41j）。")

# ======================================================================
# 8. SHAP 解釋（HGB 全樣本最終模型）
# ======================================================================
def final_hgb(R):
    cfg = Counter(tuple(sorted(c.items())) for c in R["chosen"]["hgb"]).most_common(1)[0][0]
    return fit("hgb", dict(cfg), R["X"], R["y"], R["w"]), dict(cfg)


def shap_table(R):
    model, cfg = final_hgb(R)
    expl = shap.TreeExplainer(model)
    sv = np.asarray(expl.shap_values(R["X"]))
    gap = np.abs(expl.expected_value + sv.sum(1) - model.decision_function(R["X"])).max()
    Xv = R["X"].to_numpy()
    obs = ~np.isnan(Xv)
    rows = []
    for j, f in enumerate(FEATS):
        o = obs[:, j]
        ok = o.sum() > 30 and np.std(sv[o, j]) > 0 and np.std(Xv[o, j]) > 0
        rho = pd.Series(Xv[o, j]).corr(pd.Series(sv[o, j]), method="spearman") if ok else np.nan
        rows.append(dict(特徵=f, 中文=FEAT[f][0], 領域=FEAT[f][1], 題本=book[f],
                         平均絕對SHAP_全體=np.abs(sv[:, j]).mean(), 平均絕對SHAP_有作答者=np.abs(sv[o, j]).mean(),
                         有作答n=int(o.sum()), 方向_spearman=rho))
    imp = pd.DataFrame(rows).sort_values("平均絕對SHAP_有作答者", ascending=False).reset_index(drop=True)
    imp.insert(0, "排名", np.arange(1, len(imp) + 1))
    return model, cfg, sv, imp, gap


log("\n## SHAP（HGB 最終模型；排名依「有作答該題者」的平均 |SHAP|，避免 A/B 卷題因半數遺漏被稀釋）")
shap_out = {}
for t in ["y_inc", "y_full"]:
    model, cfg, sv, imp, gap = shap_table(RES[t])
    if t == "y_inc":   # SHAP 方向（條件於其他 82 個特徵的模型內歸因）vs 調整 OR（只控制 CESD + 性別）
        o_ = or_tab.set_index("特徵")
        imp["調整OR_M1"], imp["q_M1"] = imp.特徵.map(o_.OR), imp.特徵.map(o_.q_FDR)
        same = np.sign(imp.方向_spearman) == np.sign(np.log(imp.調整OR_M1))
        imp["SHAP與OR方向"] = np.select(
            [imp.q_M1.isna(), imp.方向_spearman.isna() | (imp.方向_spearman.abs() < .1), imp.q_M1 >= .05, same],
            ["—", "SHAP 無明確方向", "OR 不顯著", "一致"], "相反")
    shap_out[t] = (model, sv, imp)
    imp.round(4).assign(目標=t).to_csv(TAB / f"shap_importance_{t}.csv", index=False, encoding="utf-8-sig")
    log(f"\n### {TARGETS[t]}（最終 HGB 設定 {cfg}；SHAP 可加性誤差 {gap:.1e}）")
    log("前 15 名：" + "、".join(f"{r.排名}.{r.中文}{'(A)' if r.題本 == 'A' else '(B)' if r.題本 == 'B' else ''}"
                              f"{'↑' if r.方向_spearman > .1 else '↓' if r.方向_spearman < -.1 else '~'}"
                              for r in imp.head(15).itertuples()))
    for f in ["cesd", "who5", "addict", "bedtime", "fomo", "late_screen", "sleep_hrs", "net_leisure", "soccomp",
              "warmth", "hostile", "cohesion"]:
        r = imp[imp.特徵 == f].iloc[0]
        log(f"  - {r.中文}: 第 {r.排名} 名 / {len(imp)}（有作答者平均|SHAP|={r.平均絕對SHAP_有作答者:.3f}）")
    if t == "y_inc":
        bad = imp[(imp.SHAP與OR方向 == "相反") | ((imp.SHAP與OR方向 == "OR 不顯著") & (imp.排名 <= 20))]
        log("  ⚠ SHAP 方向是「其他特徵固定下」的模型內條件歸因，可能受抑制效應/共線影響，不可當成實質關聯方向。"
            "SHAP 方向與調整 OR 相反、或 OR 不顯著的前 20 名特徵：" +
            "、".join(f"{r.中文}（第{r.排名}名，SHAP ρ={r.方向_spearman:+.2f}，OR {r.調整OR_M1:.2f}，q={r.q_M1:.2g}）"
                     for r in bad.itertuples()))

# 領域層級（組合 SHAP = 領域內 SHAP 加總；A 卷相關領域在 A 卷作答者中計算，家庭在 B 卷作答者中計算）
model, sv, imp = shap_out["y_inc"]
d = PRI["d"]
DOM_SAMPLE = {"睡眠與夜間作息": "has_A", "網路使用": "has_A", "個人特質與背景": "has_A", "家庭與教養": "has_B"}
dom_rows = []
for dom in dict.fromkeys(FEAT[f][1] for f in FEATS):
    js = [j for j, f in enumerate(FEATS) if FEAT[f][1] == dom and f not in ("has_A", "has_B")]
    rows_ = (d[DOM_SAMPLE[dom]] == 1).to_numpy() if dom in DOM_SAMPLE else np.ones(len(d), bool)
    dom_rows.append(dict(領域=dom, 特徵數=len(js), 計算樣本="A 卷作答者" if DOM_SAMPLE.get(dom) == "has_A" else
                         ("B 卷作答者" if dom in DOM_SAMPLE else "全體"),
                         平均絕對組合SHAP=np.abs(sv[rows_][:, js].sum(1)).mean()))
dom_tab = pd.DataFrame(dom_rows).sort_values("平均絕對組合SHAP", ascending=False)
dom_tab.round(4).to_csv(TAB / "shap_domain_importance.csv", index=False, encoding="utf-8-sig")
log("\n領域層級平均 |組合 SHAP|（新發生目標）：" + "、".join(f"{r.領域} {r.平均絕對組合SHAP:.3f}" for r in dom_tab.itertuples()))

# 移除領域 ΔAUC：拿掉整組特徵後重配適（各外折沿用原選定超參數），比較 OOF AUC —— 不受組內共線分攤功勞影響
log("\n## 移除領域後的 AUC 下降（新發生目標；各外折以原選定超參數重配適；配對學校 bootstrap）")
DROP_GROUPS = {dom: [f for f in FEATS if FEAT[f][1] == dom and f not in ("has_A", "has_B")]
               for dom in dict.fromkeys(FEAT[f][1] for f in FEATS)}
DROP_GROUPS["晚睡與深夜螢幕"] = [f for f in DROP_GROUPS["睡眠與夜間作息"] if f != "sleep_quality"]
DROP_SAMPLE = {**DOM_SAMPLE, "晚睡與深夜螢幕": "has_A"}
drop_rows = []
for gname, cols in DROP_GROUPS.items():
    keep = [f for f in FEATS if f not in cols]
    sub = (d[DROP_SAMPLE[gname]] == 1).to_numpy() if gname in DROP_SAMPLE else np.ones(len(d), bool)
    for kind in ["hgb", "lr"]:
        red = np.zeros(len(PRI["y"]))
        for (tr, te), prm in zip(PRI["folds"], PRI["chosen"][kind]):
            mm = fit(kind, prm, PRI["X"].iloc[tr][keep], PRI["y"][tr], PRI["w"][tr])
            red[te] = mm.predict_proba(PRI["X"].iloc[te][keep])[:, 1]
        full = PRI["pred"][kind]
        scopes = [("領域樣本", sub)] + ([("全體", np.ones(len(d), bool))] if not sub.all() else [])
        for scope, msk in scopes:
            ii0 = np.flatnonzero(msk)
            y, w, g, a_, b2 = PRI["y"][ii0], PRI["w"][ii0], PRI["g"][ii0], full[ii0], red[ii0]
            st = lambda ii: [wauc(y[ii], a_[ii], w[ii]) - wauc(y[ii], b2[ii], w[ii])]  # noqa: E731
            pt, bs = st(np.arange(len(y)))[0], cluster_boot(st, g)[:, 0]
            drop_rows.append(dict(移除=gname, 特徵數=len(cols), 模型=kind, 範圍=scope,
                                  評估樣本="全體" if msk.all() else
                                  ("A 卷作答者" if DROP_SAMPLE[gname] == "has_A" else "B 卷作答者"),
                                  n=len(ii0), AUC_全特徵=wauc(y, a_, w), AUC_移除後=wauc(y, b2, w), ΔAUC=pt,
                                  CI下=np.percentile(bs, 2.5), CI上=np.percentile(bs, 97.5),
                                  p_boot=min(1, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))))
drop_tab = pd.DataFrame(drop_rows)
drop_tab.round(4).to_csv(TAB / "drop_domain_auc.csv", index=False, encoding="utf-8-sig")
DROPV = drop_tab[drop_tab.範圍 == "領域樣本"]
for r in drop_tab[drop_tab.模型 == "hgb"].itertuples():
    log(f"- 移除「{r.移除}」（{r.特徵數} 個特徵；{r.評估樣本} n={r.n}）HGB AUC {r.AUC_全特徵:.3f} → {r.AUC_移除後:.3f}，"
        f"ΔAUC {r.ΔAUC:+.3f} [{r.CI下:+.3f}, {r.CI上:+.3f}], p={r.p_boot:.3f}")
lr_night = DROPV[(DROPV.模型 == "lr") & (DROPV.移除 == "晚睡與深夜螢幕")].iloc[0]
log(f"  （邏輯斯迴歸：移除「晚睡與深夜螢幕」ΔAUC {lr_night.ΔAUC:+.3f} [{lr_night.CI下:+.3f}, {lr_night.CI上:+.3f}], "
    f"p={lr_night.p_boot:.3f}，A 卷 n={lr_night.n}）")

# ======================================================================
# 9. 實際發生率（未調整）＋ 部分依賴（僅診斷：共線特徵下不可靠）
# ======================================================================
log("\n## 實際發生率（未調整，加權；CI = 學校集群 bootstrap）與部分依賴診斷（新發生目標）")
BED_EDGES = [22, 22.5, 23, 23.5, 24, 24.5]                      # 左閉右開：23:00 整點歸入 23:00–23:29
BED_LABS = ["22:00 前", "22:00–22:29", "22:30–22:59", "23:00–23:29", "23:30–23:59", "00:00–00:29", "00:30 以後"]
OBS = {"bedtime": ("has_A", BED_EDGES, BED_LABS),
       "late_screen": ("has_A", [.5], ["否", "是"]), "awake_midnight": ("has_A", [.5], ["否", "是"]),
       "fomo": ("has_A", [1.5, 2, 2.5, 3], ["<1.5", "1.5–1.99", "2–2.49", "2.5–2.99", "≥3"]),
       "addict": (None, [1.5, 2, 2.5, 3], ["<1.5", "1.5–1.99", "2–2.49", "2.5–2.99", "≥3"])}
obs_res = {}
for f, (bk, edges, labs) in OBS.items():
    rows_ = (d[bk] == 1).to_numpy() if bk else np.ones(len(d), bool)
    v_, y_, w_, g_ = d[f].to_numpy()[rows_], PRI["y"][rows_], PRI["w"][rows_], PRI["g"][rows_]
    ok_ = ~np.isnan(v_)
    v_, y_, w_, g_ = v_[ok_], y_[ok_], w_[ok_], g_[ok_]
    b_ = np.digitize(v_, edges)
    nb = len(labs)
    rate = lambda ii: [np.average(y_[ii][b_[ii] == k], weights=w_[ii][b_[ii] == k]) if (b_[ii] == k).any()  # noqa: E731
                       else np.nan for k in range(nb)]
    bs_ = cluster_boot(rate, g_, 500)
    obs_res[f] = pd.DataFrame({"區間": labs, "x": [np.average(v_[b_ == k], weights=w_[b_ == k]) for k in range(nb)],
                               "n": [int((b_ == k).sum()) for k in range(nb)], "rate": rate(np.arange(len(y_))),
                               "lo": np.nanpercentile(bs_, 2.5, axis=0), "hi": np.nanpercentile(bs_, 97.5, axis=0)})
    log(f"- {FEAT[f][0]}：" + "、".join(f"{r.區間} {r.rate:.1%} [{r.lo:.1%}, {r.hi:.1%}] (n={r.n})"
                                     for r in obs_res[f].itertuples()))
pd.concat([o.assign(特徵=f) for f, o in obs_res.items()]).round(4).to_csv(
    TAB / "observed_rate_by_bin.csv", index=False, encoding="utf-8-sig")

# 部分依賴：其他特徵固定 → 上床時間與睡眠時數(r≈-0.9)、假日上床、社會時差等被固定，形成不可能的反事實；
# 另以「拿掉共線作息特徵」重配適的 HGB 對照，顯示部分依賴對特徵組合很敏感 → 只作診斷，不作推論證據
A_rows = (d.has_A == 1).to_numpy()
XA, wA = PRI["X"][A_rows], PRI["w"][A_rows]
COLLIN = ["sleep_hrs", "wkend_bedtime", "social_jetlag", "waketime"]
keep_pd = [f for f in FEATS if f not in COLLIN]
cfg_inc = dict(Counter(tuple(sorted(c.items())) for c in PRI["chosen"]["hgb"]).most_common(1)[0][0])
model_red = fit("hgb", cfg_inc, PRI["X"][keep_pd], PRI["y"], PRI["w"])
grid_bt = np.arange(21, 26.01, .25)
pd_rows = {"bedtime": grid_bt}
for lab_, m_, Xm in [("pd_final", model, XA), ("pd_無共線作息特徵", model_red, XA[keep_pd])]:
    j = list(Xm.columns).index("bedtime")
    pd_rows[lab_] = partial_dependence(m_, Xm, [j], custom_values={j: grid_bt}, kind="average", method="brute",
                                       response_method="predict_proba", sample_weight=wA)["average"][0]
for i, m_ in enumerate(PRI["hgb_models"], 1):
    j = FEATS.index("bedtime")
    pd_rows[f"pd_fold{i}"] = partial_dependence(m_, XA, [j], custom_values={j: grid_bt}, kind="average", method="brute",
                                                response_method="predict_proba", sample_weight=wA)["average"][0]
pdp_tab = pd.DataFrame(pd_rows)
pdp_tab.round(4).to_csv(TAB / "pdp_bedtime.csv", index=False, encoding="utf-8-sig")
PDP = {k: (pdp_tab[k].iloc[0], pdp_tab[k].iloc[-1]) for k in ["pd_final", "pd_無共線作息特徵"]}
log(f"- 上床時間部分依賴（診斷）：全特徵 HGB {PDP['pd_final'][0]:.1%}（21:00）→ {PDP['pd_final'][1]:.1%}（02:00）；"
    f"拿掉睡眠時數/假日上床/社會時差/起床時間後重配適 {PDP['pd_無共線作息特徵'][0]:.1%} → {PDP['pd_無共線作息特徵'][1]:.1%}。"
    "曲線斜率取決於共線特徵是否在模型中（功勞被分攤），故部分依賴與單一特徵 SHAP 排名不能用來判斷晚睡是否重要；"
    "改以多變項 OR（第 7b 節）與移除領域 ΔAUC 判斷。")
for f in ["fomo", "addict"]:
    rows_ = A_rows if book[f] == "A" else np.ones(len(d), bool)
    Xs, ws = PRI["X"][rows_], PRI["w"][rows_]
    vals = Xs[f].dropna()
    grid = np.linspace(*np.percentile(vals, [1, 99]), 13)
    c = partial_dependence(model, Xs, [FEATS.index(f)], custom_values={FEATS.index(f): grid}, kind="average",
                           method="brute", response_method="predict_proba", sample_weight=ws)["average"][0]
    log(f"- {FEAT[f][0]}部分依賴（診斷）：{c[0]:.1%}（{grid[0]:.2f}）→ {c[-1]:.1%}（{grid[-1]:.2f}）")

# ======================================================================
# 10. 公平性：依性別、數位發展區域
# ======================================================================
log("\n## 公平性（新發生目標；門檻 = 全樣本前 15%；CI = 子群內學校 bootstrap）")
thr = {m: wquantile(PRI["pred"][m], PRI["w"], 1 - TOP) for m in ["hgb", "screener"]}
groups = {"全體": np.ones(len(d), bool), "女生": (d.female == 1).to_numpy(), "男生": (d.female == 0).to_numpy()}
for lv in ["數位發展成熟區", "數位發展潛力區", "數位發展起步區", "數位發展萌動區"]:
    groups[lv.replace("數位發展", "")] = (d.digi_level == lv).to_numpy()
fair_rows = []
for gname, msk in groups.items():
    ii0 = np.flatnonzero(msk)
    y, w, g = PRI["y"][ii0], PRI["w"][ii0], PRI["g"][ii0]
    for m in ["hgb", "screener"]:
        s = PRI["pred"][m][ii0]
        st = lambda ii: [wauc(y[ii], s[ii], w[ii]), screen_stats(y[ii], s[ii], w[ii], thr=thr[m])[0],  # noqa: E731
                         screen_stats(y[ii], s[ii], w[ii])[0]]
        bs = cluster_boot(st, g, 500)
        ss, sw = screen_stats(y, s, w, thr=thr[m]), screen_stats(y, s, w)   # 全樣本統一門檻 vs 子群內前 15%
        lo, hi = np.nanpercentile(bs, 2.5, axis=0), np.nanpercentile(bs, 97.5, axis=0)
        fair_rows.append(dict(子群=gname, 模型=MODEL_LAB[m], key=m, n=len(ii0), 學校數=len(np.unique(g)),
                              盛行率=np.average(y, weights=w), AUC=wauc(y, s, w), CI下=lo[0], CI上=hi[0],
                              標記比例=ss[4], 敏感度=ss[0], 敏感度_lo=lo[1], 敏感度_hi=hi[1], PPV=ss[1],
                              敏感度_子群內=sw[0], 敏感度_子群內_lo=lo[2], 敏感度_子群內_hi=hi[2], PPV_子群內=sw[1],
                              觀察除以預期=np.average(y, weights=w) / np.average(s, weights=w)))
fair_tab = pd.DataFrame(fair_rows)
fair_tab.round(4).to_csv(TAB / "fairness.csv", index=False, encoding="utf-8-sig")
for r in fair_tab.itertuples():
    log(f"- {r.子群} | {r.模型}: n={r.n}, 盛行率 {r.盛行率:.1%}, AUC {ci_str(r.AUC, r.CI下, r.CI上)}, "
        f"全樣本統一門檻：標記 {r.標記比例:.1%}、敏感度 {r.敏感度:.1%}、PPV {r.PPV:.1%}；"
        f"子群內前15%：敏感度 {r.敏感度_子群內:.1%}、PPV {r.PPV_子群內:.1%}；O/E {r.觀察除以預期:.2f}")

# 門檻政策的取捨（HGB）：全樣本統一前 15% / 性別內各前 15% / 各校內各前 15%（輔導室實際只能在自己學校排序）
log("\n### 門檻政策取捨（HGB；各政策皆標記約 15%；旗標於全樣本決定後固定，CI = 學校 bootstrap）")
sP, yP, wP, gP = PRI["pred"]["hgb"], PRI["y"], PRI["w"], PRI["g"]
femP = (d.female == 1).to_numpy()


def top_share(s, w, frac=TOP):
    """依分數由高到低累積權重，取累積（中點）份額 < frac 者 → 各群標記比例≈frac（小學校也不會因同分/離散而過度標記）。"""
    o = np.argsort(-s, kind="stable")
    cw = (np.cumsum(w[o]) - w[o] / 2) / w.sum()
    out = np.zeros(len(s), bool)
    out[o] = cw < frac
    return out


def policy_flags(ii, pol):
    s, w, out = sP[ii], wP[ii], np.zeros(len(ii), bool)
    if pol == "全樣本統一前 15%":
        return s >= wquantile(s, w, 1 - TOP)
    keys = femP[ii] if pol == "性別內各前 15%" else gP[ii]
    for k in np.unique(keys):
        mk = keys == k
        out[mk] = top_share(s[mk], w[mk])
    return out


def policy_stats(ii, flags):
    f, y, w, fe = flags[ii], yP[ii], wP[ii], femP[ii]
    sens = lambda m: (w * f * y)[m].sum() / (w * y)[m].sum()  # noqa: E731
    return [sens(np.ones(len(ii), bool)), (w * f * y).sum() / (w * f).sum(), (w * f).sum() / w.sum(),
            sens(fe), sens(~fe), (w * f)[fe].sum() / w[fe].sum(), (w * f)[~fe].sum() / w[~fe].sum()]


pol_rows = []
for pol in ["全樣本統一前 15%", "性別內各前 15%", "各校內各前 15%"]:
    fl = policy_flags(np.arange(len(yP)), pol)
    pt = policy_stats(np.arange(len(yP)), fl)
    bs = cluster_boot(lambda ii: policy_stats(ii, fl), gP, 500)
    lo, hi = np.nanpercentile(bs, 2.5, axis=0), np.nanpercentile(bs, 97.5, axis=0)
    pol_rows.append(dict(政策=pol, 整體敏感度=pt[0], 整體敏感度_lo=lo[0], 整體敏感度_hi=hi[0], 整體PPV=pt[1],
                         標記比例=pt[2], 女生敏感度=pt[3], 男生敏感度=pt[4], 女生標記比例=pt[5], 男生標記比例=pt[6]))
    log(f"- {pol}：整體敏感度 {pt[0]:.1%} [{lo[0]:.1%}, {hi[0]:.1%}]、PPV {pt[1]:.1%}、標記 {pt[2]:.1%}；"
        f"女生敏感度 {pt[3]:.1%}（標記 {pt[5]:.1%}）、男生敏感度 {pt[4]:.1%}（標記 {pt[6]:.1%}）")
pol_tab = pd.DataFrame(pol_rows)
pol_tab.round(4).to_csv(TAB / "threshold_policies.csv", index=False, encoding="utf-8-sig")
_fh = fair_tab[fair_tab.key == "hgb"].set_index("子群")
log(f"  → 模型分性別校準尚可（O/E 女 {_fh.loc['女生', '觀察除以預期']:.2f}、男 {_fh.loc['男生', '觀察除以預期']:.2f}；"
    f"盛行率 女 {_fh.loc['女生', '盛行率']:.1%}、男 {_fh.loc['男生', '盛行率']:.1%}），統一門檻下男生較少被標記主要反映男生基礎率較低；"
    "改用性別內門檻可讓兩性敏感度相近，但會降低女生的敏感度與整體找到的個案數 —— 這是「機會均等 vs 找到最多個案」的政策選擇，不是修正模型偏誤。")

# ======================================================================
# 11. 圖
# ======================================================================
DIVERGE = LinearSegmentedColormap.from_list("bgr", ["#2a78d6", "#c9c8c3", "#e34948"])


def beeswarm_offsets(s, rng, width=.36, nbins=70):
    h, edges = np.histogram(s, bins=nbins)
    b = np.clip(np.digitize(s, edges) - 1, 0, nbins - 1)
    return (rng.random(len(s)) * 2 - 1) * width * (h[b] / h.max()) ** .7


# --- 圖 1：SHAP 摘要（主圖） ---
rng = np.random.default_rng(SEED)
top = imp.head(15)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 7.2), gridspec_kw=dict(width_ratios=[2.5, 1], wspace=.5))
for i, f in enumerate(top.特徵[::-1]):
    j = FEATS.index(f)
    o = d[f].notna().to_numpy()
    s, v = sv[o, j], d.loc[o, f]
    rk = v.rank(pct=True).to_numpy()
    rk = (rk - rk.min()) / (rk.max() - rk.min()) if rk.max() > rk.min() else np.full(len(rk), .5)
    ax.scatter(s, i + beeswarm_offsets(s, rng), c=rk, cmap=DIVERGE, s=3, alpha=.55, lw=0, rasterized=True)
ax.axvline(0, color=INK2, lw=.8)
ax.set_yticks(range(len(top)), [label(f) for f in top.特徵[::-1]], fontsize=10)
ax.set_xlabel("SHAP 值（對預測對數勝算的貢獻；> 0 = 模型預測風險較高）")
ax.set_ylim(-.7, len(top) - .3)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color="#e6e5e0", lw=.6)
ax.set_axisbelow(True)
sm_ = plt.cm.ScalarMappable(cmap=DIVERGE)
cb = fig.colorbar(sm_, ax=ax, fraction=.025, pad=.01, ticks=[0, 1])
cb.ax.set_yticklabels(["低", "高"])
cb.set_label("該題作答值（百分位）", fontsize=9)
cb.outline.set_visible(False)
ax.set_title("A. 哪些國七特徵預示「尚未憂鬱」的學生一年後出現憂鬱或自傷意念？", loc="left")
dt = dom_tab.iloc[::-1]
ax2.barh(range(len(dt)), dt.平均絕對組合SHAP, color="#2a78d6", height=.6)
for k, (v, n) in enumerate(zip(dt.平均絕對組合SHAP, dt.計算樣本)):
    ax2.text(v + .005, k, f"{v:.2f}", va="center", fontsize=9, color=INK)
ax2.set_yticks(range(len(dt)), [f"{a}\n（{n}）" if n != "全體" else a for a, n in zip(dt.領域, dt.計算樣本)], fontsize=9.5)
ax2.set_xlabel("平均 |組合 SHAP|")
ax2.set_xlim(0, dt.平均絕對組合SHAP.max() * 1.25)
ax2.tick_params(axis="y", length=0)
ax2.spines["left"].set_visible(False)
ax2.set_title("B. 各領域整體貢獻（組合 SHAP）", loc="left")
n_inc = len(d)
fig.text(.01, .01, f"資料：TIGPS 國七(2023)→國八(2024)，國七 CESD<16 者 n={n_inc:,}；HGB 模型，SHAP 以 TreeExplainer 計算。"
         "（A卷）/（B卷）= 只有約半數學生作答的題本，僅繪出有作答者。\n"
         "SHAP 為模型內的條件預測歸因（其他特徵固定），不代表因果；相關特徵之間會分攤功勞，單一特徵的方向與排名需搭配調整後勝算比解讀。",
         fontsize=8, color=INK2)
fig.subplots_adjust(left=.2, right=.97, top=.93, bottom=.12)
fig.savefig(FIG / "fig1_shap_summary.png", dpi=200)
plt.close(fig)

# --- 圖 2：ROC / PR / 校準 / 捕捉曲線 ---
fig, axs = plt.subplots(2, 2, figsize=(10.5, 9))
y, w = PRI["y"], PRI["w"]
prev = np.average(y, weights=w)
show = ["hgb", "lr", "screener", "cesd_only"]
pt_ = perf_tab[perf_tab.目標 == "y_inc"].set_index("key")
for m in show:
    s = PRI["pred"][m]
    ls = "--" if m == "cesd_only" else "-"
    fpr, tpr, _ = roc_curve(y, s, sample_weight=w)
    axs[0, 0].plot(fpr, tpr, color=COL[m], lw=2, ls=ls, label=f"{MODEL_LAB[m]}  AUC {pt_.loc[m, 'AUC']:.3f}")
    pr, rc, _ = precision_recall_curve(y, s, sample_weight=w)
    keep = rc >= .02                                         # 召回率極低處只有個位數個案，曲線劇烈跳動，不繪
    axs[0, 1].plot(rc[keep], pr[keep], color=COL[m], lw=2, ls=ls, label=f"{MODEL_LAB[m]}  PR-AUC {pt_.loc[m, 'PR_AUC']:.3f}")
    # 校準：加權十分位
    qs = [wquantile(s, w, q) for q in np.linspace(.1, .9, 9)]
    b = np.digitize(s, qs)
    mp = [np.average(s[b == k], weights=w[b == k]) for k in range(10)]
    ob = [np.average(y[b == k], weights=w[b == k]) for k in range(10)]
    axs[1, 0].plot(mp, ob, marker="o", color=COL[m], lw=1.6, ms=5, ls=ls, label=MODEL_LAB[m])
    # 捕捉曲線
    o = np.argsort(-s)
    cw = np.cumsum(w[o]) / w.sum()
    cap = np.cumsum((w * y)[o]) / (w * y).sum()
    axs[1, 1].plot(cw * 100, cap * 100, color=COL[m], lw=2, ls=ls, label=MODEL_LAB[m])
axs[0, 0].plot([0, 1], [0, 1], color="#c9c8c3", lw=1)
axs[0, 0].set(xlabel="1 − 特異度（誤報率）", ylabel="敏感度", title="A. ROC 曲線")
axs[0, 1].axhline(prev, color="#c9c8c3", lw=1)
axs[0, 1].text(.02, prev - .045, f"隨機猜測 = 盛行率 {prev:.1%}", ha="left", fontsize=8.5, color=INK2)
axs[0, 1].set(xlabel="敏感度（召回率）", ylabel="陽性預測值 PPV", title="B. 精確率–召回率曲線", ylim=(0, 1))
lim = max(axs[1, 0].get_xlim()[1], axs[1, 0].get_ylim()[1])
axs[1, 0].plot([0, lim], [0, lim], color="#c9c8c3", lw=1)
axs[1, 0].set(xlabel="預測風險（十分位平均）", ylabel="實際發生比例", title="C. 校準（十分位）", xlim=(0, lim), ylim=(0, lim))
axs[1, 1].plot([0, 100], [0, 100], color="#c9c8c3", lw=1)
axs[1, 1].axvline(TOP * 100, color=INK2, lw=.8, ls=":")
hs = screen_stats(y, PRI["pred"]["hgb"], w)[0]
axs[1, 1].annotate(f"輔導室只約談前 15%\n就能找到 {hs:.0%} 的未來個案（HGB）", xy=(TOP * 100, hs * 100),
                   xytext=(32, 22), fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=.8))
axs[1, 1].set(xlabel="依預測風險標記的學生比例 (%)", ylabel="捕捉到的國八新發生個案 (%)", title="D. 篩檢捕捉曲線",
              xlim=(0, 100), ylim=(0, 100))
for a in axs.flat:
    a.grid(color="#eeede9", lw=.6)
    a.set_axisbelow(True)
axs[0, 0].legend(fontsize=8.5, loc="lower right", frameon=False)
axs[0, 1].legend(fontsize=8.5, loc="upper right", frameon=False)
axs[1, 0].legend(fontsize=8.5, loc="upper left", frameon=False)
fig.suptitle(f"新發生憂鬱或自傷意念風險：預測表現（學校群組 5 折交叉驗證，n={len(y):,}，加權）", fontsize=13, fontweight="bold", x=.02, ha="left")
fig.tight_layout()
fig.savefig(FIG / "fig2_performance.png", dpi=200)
plt.close(fig)

# --- 圖 3：簡短篩檢比較 ---
fig = plt.figure(figsize=(12.5, 7.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], hspace=.5, wspace=.55)
axA, axB, axC = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
order = ["cesd_only", "screener", "lr", "rf", "hgb"]
for k, m in enumerate(order):
    r = pt_.loc[m]
    axA.plot([r.AUC_lo, r.AUC_hi], [k, k], color=COL[m], lw=2.2, solid_capstyle="round")
    axA.plot(r.AUC, k, "o", color=COL[m], ms=8, mec="white", mew=1.5)
    axA.text(r.AUC_hi + .004, k, f"{r.AUC:.3f}", va="center", fontsize=9, color=INK)
axA.set_yticks(range(len(order)), [MODEL_LAB[m] for m in order], fontsize=9.5)
axA.set_xlabel("ROC-AUC（95% CI）")
axA.set_title("A. 10 題就能接近全特徵模型", loc="left")
axA.grid(axis="x", color="#eeede9", lw=.6)
axA.tick_params(axis="y", length=0)
axA.spines["left"].set_visible(False)
axA.set_xlim(axA.get_xlim()[0], axA.get_xlim()[1] + .012)
cv = curve_tab[curve_tab.目標 == "y_inc"]
axB.fill_between(cv.題數, cv.CI下, cv.CI上, color=COL["screener"], alpha=.18, lw=0)
axB.plot(cv.題數, cv.AUC, "o-", color=COL["screener"], lw=2, ms=6, mec="white", label="簡短篩檢（巢狀選題）")
axB.axhline(pt_.loc["hgb", "AUC"], color=COL["hgb"], ls="--", lw=1.4, label=f"HGB 全特徵 {pt_.loc['hgb', 'AUC']:.3f}")
axB.axhline(pt_.loc["cesd_only", "AUC"], color=COL["cesd_only"], ls=":", lw=1.4,
            label=f"僅國七 CESD 總分 {pt_.loc['cesd_only', 'AUC']:.3f}")
axB.set(xlabel="篩檢題數", ylabel="ROC-AUC", xticks=range(1, K_SCREEN + 1))
axB.set_title("B. 題數 vs 準確度", loc="left")
axB.legend(fontsize=8.5, frameon=False, loc="lower right")
axB.grid(color="#eeede9", lw=.6)
st = scr_tab[scr_tab.目標 == "y_inc"].iloc[::-1]
for k, r in enumerate(st.itertuples()):
    axC.plot([1, r.OR_每SD], [k, k], color="#c9c8c3", lw=1.5)
    axC.plot(r.OR_每SD, k, "o", color=COL["screener"], ms=8, mec="white", mew=1.5)
    up = r.OR_每SD >= 1
    axC.text(r.OR_每SD * (1.03 if up else 1 / 1.03), k, f"{r.OR_每SD:.2f}（{r.各折入選次數}/5 折）",
             va="center", ha="left" if up else "right", fontsize=8.5, color=INK)
axC.axvline(1, color=INK2, lw=.8)
axC.set_yticks(range(len(st)), [f"{r.入選順序}. {r.題目}" for r in st.itertuples()], fontsize=9.5)
axC.set_xlabel("勝算比（每增加 1 個標準差）")
axC.set_xscale("log")
lo_, hi_ = st.OR_每SD.min(), st.OR_每SD.max()
xl = (min(lo_, 1) / 1.4, max(hi_, 1) * 1.4)
axC.set_xlim(*xl)
axC.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
axC.xaxis.set_minor_locator(plt.NullLocator())
axC.set_xticks([t_ for t_ in [.5, .67, .8, 1, 1.25, 1.5, 2] if xl[0] <= t_ <= xl[1]])
axC.set_ylim(-1.2, len(st) - .5)
axC.text(.02, -.95, "← 分數越高，風險越低", transform=axC.get_yaxis_transform(), fontsize=8.5, color=INK2, ha="left")
axC.text(.98, -.95, "分數越高，風險越高 →", transform=axC.get_yaxis_transform(), fontsize=8.5, color=INK2, ha="right")
axC.tick_params(axis="y", length=0)
axC.spines["left"].set_visible(False)
axC.set_title("C. 最終 10 題（全樣本選題順序）", loc="left")
fig.suptitle("國七「10 題簡短篩檢」預測國八新發生憂鬱或自傷意念", fontsize=13, fontweight="bold", x=.02, ha="left")
fig.text(.02, .01, "題目皆為全體共同作答題；B 圖每折內獨立選題後評估（無選題洩漏）；C 圖括號為 5 個交叉驗證折中入選次數。",
         fontsize=8, color=INK2)
fig.subplots_adjust(left=.17, right=.97, top=.9, bottom=.1)
fig.savefig(FIG / "fig3_screener.png", dpi=200)
plt.close(fig)

# --- 圖 4：晚睡與深夜螢幕 —— 實際發生率、調整後勝算比、移除領域的預測增益 ---
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw=dict(width_ratios=[1.05, 1.15, 1.1], wspace=.62))
o = obs_res["bedtime"]
xs = np.arange(len(o))
a1.plot(xs, o.rate * 100, color=COL["hgb"], lw=1, alpha=.5)
a1.errorbar(xs, o.rate * 100, yerr=[(o.rate - o.lo) * 100, (o.hi - o.rate) * 100], fmt="o", color=COL["hgb"], ms=7,
            mec="white", mew=1.2, elinewidth=1.6, capsize=0)
base_A = np.average(PRI["y"][A_rows], weights=PRI["w"][A_rows]) * 100
a1.axhline(base_A, color="#c9c8c3", lw=1, ls="--")
a1.text(len(o) - .55, base_A - 2.8, f"平均 {base_A:.0f}%", ha="right", fontsize=8.5, color=INK2)
for k, r in enumerate(o.itertuples()):
    a1.text(k, r.hi * 100 + 1.2, f"{r.rate:.0%}", ha="center", fontsize=8.5, color=INK)
    a1.text(k, 3, f"n={r.n}", ha="center", fontsize=7.5, color=INK2)
a1.set_xticks(xs, [lab_.replace("–", "–\n").replace(" ", "\n") for lab_ in o.區間], fontsize=8.5)
a1.set_ylim(0, max(o.hi) * 100 + 7)
a1.set_xlabel("國七上課日上床時間")
a1.set_ylabel("國八新發生憂鬱或自傷意念 (%)")
a1.set_title("A. 越晚上床，一年後新發生率越高（未調整）", loc="left")
a1.grid(axis="y", color="#eeede9", lw=.6)
a1.set_axisbelow(True)
# B：調整後勝算比
FOREST = [("bedtime", "上課日上床時間（每晚 1 SD）"), ("awake_midnight", "午夜仍醒著（是 vs 否）"),
          ("late_screen", "23 點後以螢幕為主（是 vs 否）"), ("fomo", "錯失恐懼 FOMO（每 1 SD）"),
          ("addict", "網路成癮傾向（每 1 SD）")]
MCOL = {"M1": ("#8a8984", "M1 控制國七 CESD + 性別"), "M2": (COL["hgb"], "M2 多變項（+成癮、睡眠品質、自尊、生活滿意、WHO-5、偏好網路）"),
        "M3": (COL["screener"], "M3 去除睡眠題的 CESD（結果與控制）")}
for off, mk in zip([-.24, 0, .24], MCOL):
    for k, (f, _) in enumerate(FOREST):
        if (mk, f) not in SENS.index:
            continue
        r = SENS.loc[(mk, f)]
        a2.hlines(k + off, r.CI下, r.CI上, color=MCOL[mk][0], lw=2)
        a2.plot(r.OR, k + off, "o", color=MCOL[mk][0], ms=6.5, mec="white", mew=1, label=MCOL[mk][1] if k == 0 else None)
a2.axvline(1, color=INK2, lw=.8)
a2.set_xscale("log")
a2.set_xticks([.8, 1, 1.25, 1.5, 2, 2.5])
a2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
a2.xaxis.set_minor_locator(plt.NullLocator())
a2.set_yticks(range(len(FOREST)), [lab_ for _, lab_ in FOREST], fontsize=9.5)
a2.set_ylim(len(FOREST) - .45, -1.9)
for k, (f, _) in enumerate(FOREST):
    r = SENS.loc[("M2", f)]
    a2.text(1.02, k, f"M2: {r.OR:.2f}\np={r.p:.3f}" if r.p >= .001 else f"M2: {r.OR:.2f}\np<.001",
            transform=a2.get_yaxis_transform(), va="center", fontsize=8, color=COL["hgb"])
a2.legend(fontsize=7.6, frameon=False, loc="upper left", bbox_to_anchor=(-.02, 1.0), handletextpad=.3)
a2.set_xlabel("勝算比（95% CI，對數尺度）")
a2.set_title("B. 調整其他風險後的關聯", loc="left")
a2.tick_params(axis="y", length=0)
a2.spines["left"].set_visible(False)
a2.grid(axis="x", color="#eeede9", lw=.6)
# C：移除領域 ΔAUC
dv = DROPV.pivot(index="移除", columns="模型", values="ΔAUC")
dlo = DROPV.pivot(index="移除", columns="模型", values="CI下")
dhi = DROPV.pivot(index="移除", columns="模型", values="CI上")
smp = DROPV[DROPV.模型 == "hgb"].set_index("移除").評估樣本
order_ = dv.sort_values("hgb").index
for k, gname in enumerate(order_):
    for off, kind, mfc in [(-.15, "hgb", None), (.15, "lr", "white")]:
        c_ = COL[kind]
        a3.hlines(k + off, dlo.loc[gname, kind], dhi.loc[gname, kind], color=c_, lw=2, alpha=1 if mfc is None else .6)
        a3.plot(dv.loc[gname, kind], k + off, "o", color=c_, mfc=mfc or c_, ms=6.5, mew=1.4,
                label=(MODEL_LAB[kind].split("（")[0] if k == 0 else None))
a3.axvline(0, color=INK2, lw=.8)
a3.set_yticks(range(len(order_)), [f"{g_}{'（A卷）' if smp[g_] == 'A 卷作答者' else '（B卷）' if smp[g_] == 'B 卷作答者' else ''}"
                                   for g_ in order_], fontsize=9.5)
for k in range(len(order_)):
    if order_[k] == "晚睡與深夜螢幕":
        a3.axhspan(k - .45, k + .45, color="#fbe3d6", zorder=0, lw=0)
a3.set_xlabel("ΔAUC = 全特徵 − 移除該組後（95% CI）")
a3.set_title("C. 拿掉一組特徵，預測變差多少？", loc="left")
a3.legend(fontsize=8.5, frameon=False, loc="center right")
a3.tick_params(axis="y", length=0)
a3.spines["left"].set_visible(False)
a3.grid(axis="x", color="#eeede9", lw=.6)
m2b, m2l, m2f = (SENS.loc[("M2", f_)] for f_ in ("bedtime", "late_screen", "fomo"))
night_d = DROPV[(DROPV.模型 == "hgb") & (DROPV.移除 == "晚睡與深夜螢幕")].iloc[0]
if m2b.p < .05 and m2l.p < .05:
    head = "晚睡與深夜螢幕：調整其他風險後仍與一年後新發生風險相關" + ("，FOMO 則不再顯著" if m2f.p >= .05 else "")
else:
    head = "晚睡與深夜螢幕：調整其他風險後關聯減弱"
head += ("；但在完整預測模型中幾乎不增加預測力" if night_d.CI下 <= 0 else
         "；但對個別學生的預測增益很小" if night_d.ΔAUC < .02 else "")
fig.suptitle(head, fontsize=13, fontweight="bold", x=.01, ha="left")
fig.text(.01, .015, "資料：新發生世代（國七 CESD<16）中的 A 卷作答者（上床時間/日誌只在 A 卷）。A：加權實際發生率，95% CI 為學校集群 bootstrap。"
         "B：加權邏輯斯迴歸、學校集群穩健 SE；M2 中的睡眠品質可能是晚睡的中介，故 M2 偏保守。\n"
         "C：5 折學校群組 CV，各折沿用原選定超參數重配適；「晚睡與深夜螢幕」= 上床/起床時間、睡眠時數、假日上床、社會時差、日誌 3 項（不含睡眠品質）。"
         "關聯不等於因果。", fontsize=7.8, color=INK2)
fig.subplots_adjust(left=.05, right=.985, top=.86, bottom=.2)
fig.savefig(FIG / "fig4_night_screen.png", dpi=200)
plt.close(fig)
(FIG / "fig4_pdp.png").unlink(missing_ok=True)                 # 舊版部分依賴圖（已由上圖取代）

# --- 圖 5：公平性（AUC + 統一門檻 vs 子群內門檻的敏感度） ---
fig, (ax, axb) = plt.subplots(1, 2, figsize=(12, 5), sharey=True, gridspec_kw=dict(width_ratios=[1.2, 1], wspace=.08))
gnames = list(groups)
for off, m, mk in [(-.14, "hgb", "o"), (.14, "screener", "s")]:
    sub = fair_tab[fair_tab.key == m].set_index("子群").loc[gnames]
    yy = np.arange(len(gnames)) + off
    ax.hlines(yy, sub.CI下, sub.CI上, color=COL[m], lw=2)
    ax.plot(sub.AUC, yy, mk, color=COL[m], ms=7, mec="white", mew=1.2, label=MODEL_LAB[m], ls="none")
sub = fair_tab[fair_tab.key == "hgb"].set_index("子群").loc[gnames]
for off, col, fill, lab in [(-.14, "敏感度", True, "全樣本統一門檻（前 15%）"), (.14, "敏感度_子群內", False, "子群內各取前 15%")]:
    yy = np.arange(len(gnames)) + off
    axb.hlines(yy, sub[col + "_lo"] * 100, sub[col + "_hi"] * 100, color=COL["hgb"], lw=2, alpha=1 if fill else .5)
    axb.plot(sub[col] * 100, yy, "o", color=COL["hgb"], mfc=COL["hgb"] if fill else "white", ms=7, mew=1.6,
             ls="none", label=lab)
for k, gname in enumerate(gnames):
    axb.text(sub.loc[gname, "敏感度_hi"] * 100 + 1, k - .14, f"標記 {sub.loc[gname, '標記比例']:.0%}",
             va="center", fontsize=8, color=INK2)
ax.set_yticks(range(len(gnames)), [f"{gname}（n={int(sub.loc[gname, 'n']):,}）" for gname in gnames])
ax.invert_yaxis()
for a in (ax, axb):
    a.axhline(.5, color="#e6e5e0", lw=.8)
    a.axhline(2.5, color="#e6e5e0", lw=.8)
    a.grid(axis="x", color="#eeede9", lw=.6)
    a.tick_params(axis="y", length=0)
    a.spines["left"].set_visible(False)
ax.axvline(sub.loc["全體", "AUC"], color=COL["hgb"], lw=.8, ls=":")
ax.set_xlabel("ROC-AUC（95% CI，子群內學校 bootstrap）")
ax.set_title("A. 排序準確度（AUC）各子群相近", loc="left")
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
axb.set_xlabel("敏感度：被標記到的國八新發生個案 (%)（HGB）")
gap = sub.loc["女生", "敏感度"] - sub.loc["男生", "敏感度"]
axb.set_title("B. 全樣本統一門檻下，男生的未來個案較少被找到" if gap > .05 else "B. 各子群的敏感度", loc="left")
axb.legend(frameon=False, fontsize=8.5, loc="center right")
axb.set_xlim(0, max(sub.敏感度_hi.max(), sub.敏感度_子群內_hi.max()) * 100 + 12)
pl_ = pol_tab.set_index("政策")
fig.text(.01, .02, "門檻政策取捨（HGB，皆標記約 15%）：" + "；".join(
    f"{k} → 整體敏感度 {r.整體敏感度:.1%}（女 {r.女生敏感度:.0%}、男 {r.男生敏感度:.0%}）" for k, r in pl_.iterrows()) +
    "。\n性別內門檻讓兩性敏感度相近，但整體找到的個案較少；選哪一種是政策取捨，不是修正模型偏誤（模型分性別校準良好）。",
    fontsize=8.3, color=INK2)
fig.subplots_adjust(left=.13, right=.98, top=.9, bottom=.2)
fig.savefig(FIG / "fig5_fairness.png", dpi=200)
plt.close(fig)

# ======================================================================
# 12. 重點摘要（由本次結果自動產生）→ results.md
# ======================================================================
pi = perf_tab[perf_tab.目標 == "y_inc"].set_index("key")
pc = perf_tab[perf_tab.目標 == "y_inc_cesd"].set_index("key")
ps = perf_tab[perf_tab.目標 == "y_inc_strict"].set_index("key")
pf = perf_tab[perf_tab.目標 == "y_full"].set_index("key")
prv = pd.DataFrame(prev_rows).set_index("結果")
dl = delta_tab.set_index(["目標", "比較"])
imp_i = shap_out["y_inc"][2].set_index("特徵")
orr = or_tab.set_index("特徵")
inc_ = pd.DataFrame(inc_rows).set_index("key")
incA, incN, incF, incB = inc_.loc["A"], inc_.loc["A_night"], inc_.loc["A_fomo"], inc_.loc["B"]
fh = fair_tab[fair_tab.key == "hgb"].set_index("子群")
pol = pol_tab.set_index("政策")
s15 = thr_tab[(thr_tab.模型 == MODEL_LAB["hgb"]) & np.isclose(thr_tab.標記比例, TOP)].iloc[0]
dnight = DROPV[(DROPV.模型 == "hgb") & (DROPV.移除 == "晚睡與深夜螢幕")].iloc[0]
dsleep = DROPV[(DROPV.模型 == "hgb") & (DROPV.移除 == "睡眠與夜間作息")].iloc[0]
gap_hs = dl.loc[("y_inc", f"{MODEL_LAB['hgb']} − {MODEL_LAB['screener']}"), "ΔAUC"]
disc = imp_i[imp_i.SHAP與OR方向 == "相反"]


def dget(t, a, b):
    r = dl.loc[(t, f"{MODEL_LAB[a]} − {MODEL_LAB[b]}")]
    return f"{r.ΔAUC:+.3f} [{r.CI下:+.3f}, {r.CI上:+.3f}]"


def ci(r, k):
    return f"{r[k]:.3f} [{r[k + '_lo']:.3f}, {r[k + '_hi']:.3f}]"


def orf(f):
    return f"OR {orr.loc[f, 'OR']:.2f} [{orr.loc[f, 'CI下']:.2f}, {orr.loc[f, 'CI上']:.2f}], q={orr.loc[f, 'q_FDR']:.2g}"


def ors(mk, f):
    r = SENS.loc[(mk, f)]
    return f"OR {r.OR:.2f} [{r.CI下:.2f}, {r.CI上:.2f}], {pfmt(r.p)}"


def prev_s(t):
    r = prv.loc[TARGETS[t]]
    return f"{r.加權盛行率:.1%} [{r.CI下:.1%}, {r.CI上:.1%}]"


def dnum(r):
    return f"{r.ΔAUC:+.3f} [{r.CI下:+.3f}, {r.CI上:+.3f}], {pfmt(r.p_boot)}"


m2 = {f: SENS.loc[("M2", f)] for f in ["bedtime", "late_screen", "awake_midnight", "fomo"]}
indep = [FEAT[f][0] for f in ["bedtime", "late_screen", "awake_midnight"] if m2[f].p < .05]
night_head = (("關聯穩健且獨立於成癮/睡眠品質/幸福感" if len(indep) >= 2 else "關聯多被其他因子解釋") + "；預測上，"
              + ("對 10 題篩檢有小幅但顯著的增益" if incA.p_boot < .05 else "對 10 題篩檢無顯著增益")
              + ("，在 83 特徵完整模型中則無額外增益" if dnight.CI下 <= 0 else "，在完整模型中仍有小幅增益"))
weak_dir = imp_i[(imp_i.SHAP與OR方向 == "OR 不顯著") & (imp_i.排名 <= 20)]
m4ok = all((SENS.loc[("M4", f)].p < .05) == (orr.loc[f, "p"] < .05) for f in ["bedtime", "late_screen", "hostile", "warmth"])
strict_ok = abs(ps.loc["hgb", "AUC"] - pi.loc["hgb", "AUC"]) < .03
SUM = [
    "# 國七 → 國八 憂鬱或自傷意念風險早期預警模型（scripts/05_risk_prediction.py）", "",
    "## 重點摘要（由本次執行結果自動產生）",
    f"- **主目標＝新發生憂鬱或自傷意念風險**：國七 CESD<16 的 {len(PRI['y']):,} 人中，國八有 {prev_s('y_inc')}（加權）"
    f"出現 CESD≥16 或近兩週自傷/求死意念。注意：其中 {COMP['w_sh_only']:.0%}（{COMP['n_sh_only']:,}/{COMP['n_case']:,}）"
    f"國八 CESD<16、只因自傷/求死意念題入列，{COMP['w_weak']:.0%} 只靠單一題的最低頻率（1–2 天）→ 不宜簡稱「憂鬱高風險」。"
    f"共同主要結果「新發生憂鬱症狀」（只看 CESD≥16）{prev_s('y_inc_cesd')}；嚴格自傷門檻 {prev_s('y_inc_strict')}；"
    f"全樣本國八風險 {prev_s('y_full')}。",
    f"- **預測力中等**：HGB AUC {ci(pi.loc['hgb'], 'AUC')}、邏輯斯迴歸 {ci(pi.loc['lr'], 'AUC')}、"
    f"隨機森林 {ci(pi.loc['rf'], 'AUC')}；只用國七 CESD 總分 {ci(pi.loc['cesd_only'], 'AUC')}。"
    f"全特徵 HGB 比只看 CESD 多 ΔAUC {dget('y_inc', 'hgb', 'cesd_only')}；HGB − 邏輯斯迴歸 ΔAUC {dget('y_inc', 'hgb', 'lr')}"
    " → 非線性 ML 沒有帶來額外好處，可解釋的線性模型已足夠。"
    f"共同主要結果（只看 CESD）：HGB {ci(pc.loc['hgb'], 'AUC')}、邏輯斯 {ci(pc.loc['lr'], 'AUC')}、"
    f"10 題篩檢 {ci(pc.loc['screener'], 'AUC')}、只看 CESD {ci(pc.loc['cesd_only'], 'AUC')}；"
    f"嚴格自傷門檻：HGB {ci(ps.loc['hgb'], 'AUC')}、邏輯斯 {ci(ps.loc['lr'], 'AUC')}"
    + (" → 結論不依賴寬鬆的自傷門檻。" if strict_ok else "。"),
    f"- **篩檢實用性**：標記預測風險前 15% 的學生，可找到 {s15.敏感度:.0%} 的國八新發生個案，PPV {s15.PPV:.0%}"
    f"（盛行率的 {s15.提升倍數:.1f} 倍）；HGB 校準良好（校準斜率 {pi.loc['hgb', '校準斜率']:.2f}）。"
    f"邏輯斯迴歸以 AUC 選到強 L2 收縮（C 多為 {Counter(RES['y_inc']['chosen']['lr']).most_common(1)[0][0]:g}），"
    f"排序不受影響但機率被壓縮（校準斜率 {pi.loc['lr', '校準斜率']:.2f}），若要報絕對風險需再校準。",
    f"- **10 題簡短篩檢**（每折內巢狀選題）AUC {ci(pi.loc['screener'], 'AUC')}；HGB − 10 題 ΔAUC {dget('y_inc', 'hgb', 'screener')}。"
    "題目：" + "、".join(SCREEN_POOL[i] for i in PRI["scr_final"]) + "。",
    "- **SHAP（模型內條件歸因，非因果）**：前五名 " + "、".join(r.中文 for r in shap_out["y_inc"][2].head(5).itertuples()) +
    f"；國七（次閾值）憂鬱症狀仍是最大單一因子，網路成癮第 {imp_i.loc['addict', '排名']} 名。領域組合 SHAP："
    + "、".join(f"{r.領域} {r.平均絕對組合SHAP:.2f}" for r in dom_tab.itertuples()) +
    "。單一特徵的 SHAP 排名/方向在相關特徵之間會分攤或翻轉"
    + ("（" + "、".join(f"{r.中文} SHAP ρ={r.方向_spearman:+.2f} 但 OR {r.調整OR_M1:.2f}" for r in disc.itertuples()) + "）"
       if len(disc) else "") +
    ("；" + "、".join(f"{r.中文}（第{r.排名}名）SHAP ρ={r.方向_spearman:+.2f} 但調整 OR {r.調整OR_M1:.2f} 不顯著"
                     for r in weak_dir.itertuples()) if len(weak_dir) else "") +
    "，皆屬模型內的條件/抑制效應，不作為實質關聯方向的證據。",
    f"- **晚睡與深夜螢幕：{night_head}**。控制國七 CESD 與性別（M1）：上床時間每晚 1 SD {orf('bedtime')}；"
    f"23 點後以螢幕為主 {orf('late_screen')}；午夜仍醒著 {orf('awake_midnight')}；FOMO 每 SD {orf('fomo')}。"
    f"再控制網路成癮、睡眠品質、自尊、生活滿意、WHO-5、偏好網路（M2，A 卷 n={int(m2['bedtime'].n):,}）："
    f"上床時間 {ors('M2', 'bedtime')}；23 點後螢幕 {ors('M2', 'late_screen')}；午夜仍醒著 {ors('M2', 'awake_midnight')}；"
    f"FOMO {ors('M2', 'fomo')}"
    + (f" → {'、'.join(indep)} 仍獨立相關" if indep else " → 晚睡相關變項的關聯被其他因子解釋")
    + ("，FOMO 的關聯則被其他因子吸收。" if m2["fomo"].p >= .05 else "，FOMO 亦仍相關。")
    + f"去除 CESD 睡眠題（M3）結果一致：上床時間 {ors('M3', 'bedtime')}、23 點後螢幕 {ors('M3', 'late_screen')}；"
    f"限主卷作答者（M4）：上床時間 {ors('M4', 'bedtime')}、23 點後螢幕 {ors('M4', 'late_screen')}。"
    f"預測面：上床時間 + FOMO + 23 點後螢幕加入 10 題篩檢 ΔAUC {dnum(incA)}（約為 HGB 與篩檢差距 {gap_hs:.3f} 的 "
    f"{incA.ΔAUC / gap_hs:.0%}）；只加晚睡三項 {dnum(incN)}；只加 FOMO {dnum(incF)}。"
    f"從 HGB 移除整組「晚睡與深夜螢幕」ΔAUC {dnum(dnight)}，移除整個睡眠領域 {dnum(dsleep)}（A 卷 n={int(dnight.n):,}）。"
    f"未調整發生率由 23:00 前約 {obs_res['bedtime'].rate.iloc[:3].min():.0%}–{obs_res['bedtime'].rate.iloc[:3].max():.0%} "
    f"升到 00:30 以後 {obs_res['bedtime'].rate.iloc[-1]:.0%}。上床時間與睡眠時數 r={CORR['bedtime~sleep_hrs']:.2f}，"
    f"HGB 部分依賴（{PDP['pd_final'][0]:.1%}→{PDP['pd_final'][1]:.1%}）與單一特徵 SHAP 排名在此共線下不可靠"
    f"（拿掉共線作息特徵後部分依賴變為 {PDP['pd_無共線作息特徵'][0]:.1%}→{PDP['pd_無共線作息特徵'][1]:.1%}），不作證據。"
    "以上皆為關聯，非因果。",
    f"- **教養**：家長敵意/冷漠（每 SD {orf('hostile')}）、家長溫暖（每 SD {orf('warmth')}）、"
    f"數位教養責罵衝突（每 SD {orf('par_conflict')}）與風險相關；限制（{orf('par_restrict')}）與監看（{orf('par_monitor')}）"
    "未達顯著（CI 仍容許小幅效果；只控制 CESD 與性別，屬關聯，不能據此推論「限制沒用」）。"
    f"限主卷作答者：敵意 {ors('M4', 'hostile')}、溫暖 {ors('M4', 'warmth')}。"
    f"教養題組加入篩檢 ΔAUC {dnum(incB)}（B 卷 n={int(incB.n):,}）。",
    f"- **公平性**：AUC 女 {fh.loc['女生', 'AUC']:.3f}、男 {fh.loc['男生', 'AUC']:.3f}；數位發展區 "
    + "、".join(f"{g_} {fh.loc[g_, 'AUC']:.2f}" for g_ in ["成熟區", "潛力區", "起步區", "萌動區"]) +
    f"（萌動區 n={int(fh.loc['萌動區', 'n'])}，CI 很寬）。用全樣本統一門檻（全國樣本一個切點）時，女生被標記 {fh.loc['女生', '標記比例']:.0%}、"
    f"敏感度 {fh.loc['女生', '敏感度']:.0%}；男生只被標記 {fh.loc['男生', '標記比例']:.0%}、敏感度 {fh.loc['男生', '敏感度']:.0%}"
    f"（O/E 女 {fh.loc['女生', '觀察除以預期']:.2f}、男 {fh.loc['男生', '觀察除以預期']:.2f}；男生盛行率較低 "
    f"{fh.loc['男生', '盛行率']:.0%} vs {fh.loc['女生', '盛行率']:.0%}）。"
    f"改為性別內各取前 15%：女 {pol.loc['性別內各前 15%', '女生敏感度']:.0%}／男 {pol.loc['性別內各前 15%', '男生敏感度']:.0%}，"
    f"但整體敏感度由 {pol.loc['全樣本統一前 15%', '整體敏感度']:.1%} 降為 {pol.loc['性別內各前 15%', '整體敏感度']:.1%}"
    f"（PPV {pol.loc['全樣本統一前 15%', '整體PPV']:.0%}→{pol.loc['性別內各前 15%', '整體PPV']:.0%}）；"
    f"若改為各校內各取前 15%（較接近輔導室實務），整體敏感度 {pol.loc['各校內各前 15%', '整體敏感度']:.1%}，"
    f"性別差距依舊（女 {pol.loc['各校內各前 15%', '女生敏感度']:.0%}／男 {pol.loc['各校內各前 15%', '男生敏感度']:.0%}）。"
    "這是「兩性機會均等 vs 找到最多個案」的政策選擇，而非修正模型偏誤。",
    "- **限制**：預測 ≠ 因果；新發生目標中的自傷意念在國七未測量（不一定是「新」發生）；"
    f"主卷以學校為單位分派（非個人隨機），且 A/B 卷作答者中約 {BOOKV['volA'] / BOOKV['nA']:.0%}／{BOOKV['volB'] / BOOKV['nB']:.0%} 為自願加填附加卷者"
    "（非隨機，女生比例較高）" + ("，限主卷的敏感度分析結論一致；" if m4ok else "，限主卷的敏感度分析有部分結論改變（見第 7b 節）；") +
    "AUC ≈ 0.72 只適合作為「輔導關懷的優先順序」參考，不可作為診斷或標籤。", "",
]
log(f"\n完成，總耗時 {time.time() - T0:.0f}s。圖：{', '.join(sorted(x.name for x in FIG.glob('*.png')))}")
(OUT / "results.md").write_text("\n".join(SUM + LOG[1:]) + "\n", encoding="utf-8")
print("\n".join(SUM))
