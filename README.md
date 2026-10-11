# 2026 資料科學漫步：TIGPS 數位世代青少年

使用資料：臺灣數位世代青少年的成長歷程追蹤研究（TIGPS）學生問卷
- 第一波（國七，2023）D00262：https://doi.org/10.6141/TW-SRDA-D00262-2
- 第二波（國八，2024）D00268：https://doi.org/10.6141/TW-SRDA-D00268-1

原始資料依 SRDA 使用規範不隨程式碼散布，請以一般會員身分自 SRDA 下載後，解壓至 `data/raw/D00262/`、`data/raw/D00268/`。

## 重現步驟
Python 3.12.10；套件版本固定於 `requirements.txt`。依序執行（隨機種子：02、03、05 為 20261019，04 為 2026）：
```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -I scripts/01_eda.py              # 初步探索
.venv/Scripts/python -I scripts/02_evening_typology.py # 國七晚間類型學（約 15 分鐘）
.venv/Scripts/python -I scripts/03_sleep_mediation.py  # 睡眠中介、兩波交叉延宕
.venv/Scripts/python -I scripts/04_digital_parenting.py # 數位教養
.venv/Scripts/python -I scripts/05_risk_prediction.py  # 早期預警模型（約 7 分鐘）
.venv/Scripts/python -I scripts/06_report_figures.py   # 報告圖 1–4、表 1
.venv/Scripts/python -I scripts/07_build_report.py     # report/report.html → report.pdf、pages/*.png（需 Microsoft Edge）
```

## 結構
- `src/tigps/data.py`：讀檔、遺漏值、量表建構、上床時間 12/24 小時制校正、兩波串接（以 `nstudent_id`）
- `scripts/01_eda.py`：初步探索，輸出 `outputs/figures`、`outputs/tables`
- `scripts/02_evening_typology.py`：晚間日誌序列分群（LCA／PAM／Ward／k-modes）與類型 → 國八結果，輸出 `outputs/typology`
- `scripts/03_sleep_mediation.py`：半縱貫中介、CLPM、E-value、日誌螢幕特異性，輸出 `outputs/mediation`
- `scripts/04_digital_parenting.py`：數位教養構面、同期／前瞻模型、時序負對照、調節與反向檢驗，輸出 `outputs/parenting`
- `scripts/05_risk_prediction.py`：學校分組巢狀交叉驗證的預警模型、10 題篩檢、SHAP、公平性，輸出 `outputs/prediction`
- `scripts/06_report_figures.py`：由上述 CSV 重繪報告圖表，輸出 `report/figures`、`report/tables`
- `scripts/07_build_report.py`：以 Edge headless 將 `report/report.html` 輸出為 PDF 並轉成逐頁 PNG

## 資料處理重點
- 負值代碼（-999 跳答、-9 未填等）皆視為遺漏。
- W1 學生問卷分 A/B 卷：晚間作息、睡眠、上網時間、FOMO 僅 A 卷（n≈4,700）；數位教養僅 B 卷（n≈4,800）。W2 無分卷。
- CESD-R 以兩波共同 12 題計分（0–48）；W2 額外 2 題自傷/求死意念另行處理。分析睡眠時使用排除睡眠題的 11 題版本。
- W1 上床時間混用 12/24 小時制：1–5 點視為凌晨、6–12 點視為晚上，合理範圍 20:00–05:00。
- 加權使用 `HOUWGT`（相對權重），標準誤以學校叢集校正。
