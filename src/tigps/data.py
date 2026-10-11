"""TIGPS 學生問卷：讀檔、遺漏值處理、量表建構與兩波串接。

所有負值代碼（-999 跳答、-99 不合理、-9 未填、-8 拒答 ...）一律視為遺漏。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
W1_SAV = RAW / "D00262" / "標準版" / "01_學生" / "TIGPSw1_s.sav"
W2_SAV = RAW / "D00268" / "標準版" / "01_學生" / "TIGPSw2_s.sav"

DESIGN = ["nstudent_id", "nschool_id", "school_class", "school_type", "digi_level",
          "JKZONE", "JKREP", "TOTWGT", "HOUWGT"]

# CESD-R：兩波共同的 12 題（W2 另有 bs56e 想傷害自己、bs56i 寧願死亡兩題）
CESD_W1 = [f"as72{c}" for c in "abcdefghijkl"]
CESD_W2 = [f"bs56{c}" for c in "abcdfghjklmn"]
CESD_SLEEP_ITEM = {"w1": "as72b", "w2": "bs56b"}  # 分析睡眠時須排除，避免題目重疊
SELFHARM_W2 = ["bs56e", "bs56i"]

ADDICT_W1 = [f"as41{c}" for c in "abcdefghij"]
ADDICT_W2 = [f"bs28{c}" for c in "abcdefghij"]
WHO5_W1 = [f"as73{c}" for c in "abcde"]
WHO5_W2 = [f"bs58{c}" for c in "abcde"]
FOMO_W1 = [f"as39{c}" for c in "abcd"]
FOMO_W2 = [f"bs27{c}" for c in "abcd"]
SOCCOMP_W1 = [f"as40{c}" for c in "abc"]   # 前三題為「向上比較」
SOCCOMP_W2 = [f"bs26{c}" for c in "abc"]

# 平日上網時間（1=沒有 ... 12=5 小時以上）→ 組中點小時數
NET_HOURS = {1: 0, 2: .25, 3: .75, 4: 1.25, 5: 1.75, 6: 2.25, 7: 2.75, 8: 3.25,
             9: 3.75, 10: 4.25, 11: 4.75, 12: 5.5}
NET_W1 = dict(homework="as35a", learning="as35b", game="as35c", video="as35d",
              chat="as35e", browse="as35f")
NET_W2 = dict(homework="bs21a", learning="bs21b", game="bs21c", video="bs21d",
              chat="bs21e", browse="bs21f")

# 晚間逐時主要活動（僅 W1，A 卷）
DIARY_SLOTS = {"18": "as70a", "19": "as70c", "20": "as70e", "21": "as70g",
               "22": "as70i", "23": "as70k", "00": "as70m", "01": "as70o"}
# 數位教養（父、母各題取兩位家長可用值的平均；5=不適用 → 遺漏）
# W2 每位家長 12 題 bs20a?(父) / bs20b?(母)
PAR_W2 = {
    "restrict": "ab",    # 規定何時可上網、限制時間長短
    "active": "cdf",     # 聊網路趣聞、陪查資料寫作業、討論使用心得
    "encourage": "e",    # 鼓勵上網
    "worry": "gh",       # 擔心健康、擔心課業
    "monitor": "ij",     # 問在網上做什麼、查看瀏覽紀錄
    "conflict": "kl",    # 因上網責罵、因網購起衝突
}
# W1（僅 B 卷）每位家長 18 題 as28?(父) / as29?(母)
PAR_W1 = {
    "restrict": "abc",   # 規定時間、禁止某些網站、限制時間長短
    "active": "dfh",     # 聊網路趣聞、陪查資料寫作業、討論使用心得（e 一起玩遊戲另計）
    "encourage": "g",
    "worry": "jkl",
    "monitor": "mo",     # 問在網上做什麼、查看瀏覽紀錄（n 知道網友另計）
    "conflict": "pqr",   # 碎碎念、責罵、網購衝突
}
# 一般親子關係（作為數位教養的混淆控制）
WARMTH = {"w1": [f"as12{c}" for c in "bcd"], "w2": [f"bs6{c}" for c in "bcd"]}
HOSTILE = {"w1": [f"as12{c}" for c in "aef"], "w2": [f"bs6{c}" for c in "aef"]}
COHESION = {"w1": [f"as9{c}" for c in "abcdef"], "w2": [f"bs5{c}" for c in "abcdef"]}

DIARY_GROUPS = {  # 21 類活動 → 7 大類
    1: "補習/課輔", 2: "寫作業讀書", 3: "寫作業讀書", 4: "運動社團才藝",
    5: "吃飯家事生活", 6: "吃飯家事生活", 7: "吃飯家事生活", 9: "吃飯家事生活",
    10: "吃飯家事生活", 12: "吃飯家事生活", 13: "吃飯家事生活", 20: "運動社團才藝",
    8: "睡覺", 11: "與家人朋友聊天",
    14: "螢幕：遊戲/電視", 15: "螢幕：影音", 16: "螢幕：社群聊天", 17: "螢幕：社群聊天",
    18: "螢幕：社群聊天", 19: "螢幕：影音", 21: "其他",
}


def read_wave(path: Path) -> tuple[pd.DataFrame, pyreadstat.metadata_container]:
    df, meta = pyreadstat.read_sav(str(path))
    num = df.select_dtypes("number").columns.difference(DESIGN)
    df[num] = df[num].where(df[num] >= 0)
    return df, meta


def scale_mean(df: pd.DataFrame, items: list[str], min_frac: float = .8) -> pd.Series:
    """題目平均；作答題數不足 min_frac 者設為遺漏。"""
    x = df[items]
    ok = x.notna().sum(axis=1) >= np.ceil(len(items) * min_frac)
    return x.mean(axis=1).where(ok)


def parenting(df: pd.DataFrame, father: str, mother: str, mapping: dict) -> pd.DataFrame:
    """各構面 = 父、母各自題目平均後，再取有作答家長的平均。"""
    out = {}
    for name, letters in mapping.items():
        f = df[[father + c for c in letters]].where(lambda x: x <= 4).mean(axis=1)
        m = df[[mother + c for c in letters]].where(lambda x: x <= 4).mean(axis=1)
        out[f"par_{name}"] = pd.concat([f, m], axis=1).mean(axis=1)
    return pd.DataFrame(out, index=df.index)


def clock_bedtime(h: pd.Series, m: pd.Series) -> pd.Series:
    """上床時間 → 以 0 點為 24 的連續小時（如 23.5、25.0）。

    W1 學生混用 12/24 小時制（9、10、11 點多為晚上），故 6–12 視為晚上 +12，
    1–5 視為凌晨 +24；合理範圍限定 20:00–05:00。
    """
    h = h.copy()
    h = np.where((h >= 1) & (h <= 5), h + 24, np.where((h >= 6) & (h <= 12), h + 12, h))
    t = pd.Series(h, index=m.index) + m.fillna(0) / 60
    return t.where((t >= 20) & (t <= 29))


def clock_waketime(h: pd.Series, m: pd.Series) -> pd.Series:
    t = h + m.fillna(0) / 60
    return t.where((t >= 4) & (t <= 10))


def build_w1(df: pd.DataFrame) -> pd.DataFrame:
    out = df[DESIGN].copy()
    out["booklet"] = df["response"].map({1: "A+B", 2: "B+A", 3: "A", 4: "B", 5: "A"})
    out["has_A"] = df["response"].isin([1, 2, 3, 5])
    out["has_B"] = df["response"].isin([1, 2, 4])
    out["female"] = (df["as1"] == 1).astype(float).where(df["as1"].notna())
    out["fam_econ"] = df["as10"]                       # 1 很不富裕 – 4 很富裕
    out["grade_rank"] = df["as20"]                     # 1 全班前五 – 5 三十名後
    out["life_sat"] = df["as62"]
    out["cesd"] = scale_mean(df, CESD_W1) * 12         # 0–48
    out["cesd_nosleep"] = scale_mean(df, [c for c in CESD_W1 if c != CESD_SLEEP_ITEM["w1"]]) * 11
    out["addict"] = scale_mean(df, ADDICT_W1)          # 1–4
    out["who5"] = scale_mean(df, WHO5_W1) * 5          # 0–20
    out["fomo"] = scale_mean(df, FOMO_W1)
    out["soccomp"] = scale_mean(df, SOCCOMP_W1)
    for k, c in NET_W1.items():
        out[f"net_{k}"] = df[c].map(NET_HOURS)
    out["net_leisure"] = out[[f"net_{k}" for k in ("game", "video", "chat", "browse")]].sum(axis=1, min_count=4)
    out["bedtime"] = clock_bedtime(df["as68a1"], df["as68a2"])
    out["waketime"] = clock_waketime(df["as68b1"], df["as68b2"])
    out["sleep_hrs"] = (out["waketime"] + 24 - out["bedtime"]).where(lambda s: s.between(3, 13))
    out["sleep_quality"] = df["as68e"]
    for hr, c in DIARY_SLOTS.items():
        out[f"diary_{hr}"] = df[c]
    out["bully_victim"] = (df["as51"] == 1).astype(float).where(df["as51"].notna())
    out["cyber_victim"] = (df["as53"] == 1).astype(float).where(df["as53"].notna())
    out["parents_married_cohab"] = (df["as3"] == 1).astype(float).where(df["as3"].notna())
    out["cram"] = (df["as22"] == 1).astype(float).where(df["as22"].notna())
    out["warmth"] = scale_mean(df, WARMTH["w1"])
    out["hostile"] = scale_mean(df, HOSTILE["w1"])
    out["cohesion"] = scale_mean(df, COHESION["w1"])
    par = parenting(df, "as28", "as29", PAR_W1)
    par.loc[~out["has_B"]] = np.nan
    out[par.columns] = par
    return out


def build_w2(df: pd.DataFrame) -> pd.DataFrame:
    out = df[DESIGN].copy()
    out["grade_rank"] = df["bs15"]
    out["life_sat"] = df["bs50"]
    out["cesd"] = scale_mean(df, CESD_W2) * 12
    out["cesd_nosleep"] = scale_mean(df, [c for c in CESD_W2 if c != CESD_SLEEP_ITEM["w2"]]) * 11
    sh = df[SELFHARM_W2]
    out["selfharm"] = (sh >= 1).any(axis=1).astype(float).where(sh.notna().any(axis=1))
    out["addict"] = scale_mean(df, ADDICT_W2)
    out["who5"] = scale_mean(df, WHO5_W2) * 5
    out["fomo"] = scale_mean(df, FOMO_W2)
    out["soccomp"] = scale_mean(df, SOCCOMP_W2)
    for k, c in NET_W2.items():
        out[f"net_{k}"] = df[c].map(NET_HOURS)
    out["net_leisure"] = out[[f"net_{k}" for k in ("game", "video", "chat", "browse")]].sum(axis=1, min_count=4)
    out["bedtime"] = clock_bedtime(df["bs60ah"], df["bs60am"])
    out["waketime"] = clock_waketime(df["bs60bh"], df["bs60bm"])
    out["sleep_hrs"] = (out["waketime"] + 24 - out["bedtime"]).where(lambda s: s.between(3, 13))
    out["sleep_quality"] = df["bs60e"]
    out["bully_victim"] = (df["bs34"] == 1).astype(float).where(df["bs34"].notna())
    out["cyber_victim"] = (df["bs38"] == 1).astype(float).where(df["bs38"].notna())
    out["female"] = (df["bs1"] == 1).astype(float).where(df["bs1"].notna())
    out["subj_ses"] = df["bs3"]                        # 主觀社會階層 1–10
    out["parents_married_cohab"] = (df["bs2"] == 1).astype(float).where(df["bs2"].notna())
    out["cram"] = (df["bs17"] == 1).astype(float).where(df["bs17"].notna())
    out["warmth"] = scale_mean(df, WARMTH["w2"])
    out["hostile"] = scale_mean(df, HOSTILE["w2"])
    out["cohesion"] = scale_mean(df, COHESION["w2"])
    out[list(f"par_{k}" for k in PAR_W2)] = parenting(df, "bs20a", "bs20b", PAR_W2)
    out["study_selfreg"] = scale_mean(df, ["bs45a", "bs45b"])  # 讀書時關掉干擾、先完成作業才玩
    return out


def load_panel() -> pd.DataFrame:
    """寬格式：W1 變項後綴 _1、W2 後綴 _2，以 nstudent_id 內部串接。"""
    w1 = build_w1(read_wave(W1_SAV)[0])
    w2 = build_w2(read_wave(W2_SAV)[0])
    keep = ["nstudent_id", "nschool_id", "school_class", "school_type", "digi_level",
            "JKZONE", "JKREP", "TOTWGT", "HOUWGT"]
    a = w1.set_index("nstudent_id").add_suffix("_1")
    b = w2.drop(columns=keep[1:]).set_index("nstudent_id").add_suffix("_2")
    panel = a.join(b, how="inner")
    panel = panel.rename(columns={f"{k}_1": k for k in keep[1:]})
    return panel.reset_index()
