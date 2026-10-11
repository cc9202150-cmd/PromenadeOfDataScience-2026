"""報告輸出：report/report.html（＋style.css、figures/）→ report/report.pdf → report/pages/page_N.png。

步驟
  1. 以 Microsoft Edge（Chromium）headless 模式列印 HTML 為 A4 PDF（--no-pdf-header-footer；頁碼由 CSS @page 產生）。
     Edge 路徑可用環境變數 EDGE_PATH 覆寫；使用暫存 user-data-dir，避免與已開啟的 Edge 衝突。
  2. 以 pypdfium2 將每頁轉成 PNG（預設 110 dpi），供逐頁目視檢查。
  3. 檢查：總頁數須在 3–5 頁（含首頁）；摘要字數須在 100–200 字（不含空白）。不符時以非零代碼結束。

執行：.venv/Scripts/python -I scripts/07_build_report.py [--dpi 110]
前置：先執行 scripts/06_report_figures.py 產生 report/figures/*.png。
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from html.parser import HTMLParser
from pathlib import Path

import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
HTML, PDF, PAGES = REPORT / "report.html", REPORT / "report.pdf", REPORT / "pages"
EDGE_CANDIDATES = [
    os.environ.get("EDGE_PATH", ""),
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
PAGE_MIN, PAGE_MAX = 3, 5
ABS_MIN, ABS_MAX = 100, 200

sys.stdout.reconfigure(encoding="utf-8")


def find_edge() -> str:
    for c in EDGE_CANDIDATES:
        if c and Path(c).is_file():
            return c
    found = shutil.which("msedge") or shutil.which("microsoft-edge")
    if found:
        return found
    sys.exit("找不到 Microsoft Edge；請設定環境變數 EDGE_PATH。")


def print_pdf(edge: str) -> None:
    for f in ["style.css", "figures/fig1_late_sleep_risk.png", "figures/fig2_evening_types.png",
              "figures/fig3_clpm.png", "figures/fig4_not_only_screens.png", "figures/fig5_parenting.png"]:
        if not (REPORT / f).is_file():
            sys.exit(f"缺少 {REPORT / f}；請先執行 scripts/06_report_figures.py。")
    if PDF.exists():
        PDF.unlink()
    # Windows 上 msedge.exe 啟動器會把工作交給子程序後立即返回，故以輪詢等待 PDF 寫完（大小連續穩定）。
    prof = tempfile.mkdtemp(prefix="edge_pdf_")
    try:
        cmd = [edge, "--headless", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
               "--disable-extensions", f"--user-data-dir={prof}", "--no-pdf-header-footer",
               "--virtual-time-budget=10000", "--run-all-compositor-stages-before-draw",
               f"--print-to-pdf={PDF}", HTML.resolve().as_uri()]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
        t0, last, stable = time.time(), -1, 0
        while time.time() - t0 < 120:
            size = PDF.stat().st_size if PDF.is_file() else -1
            stable = stable + 1 if (size > 0 and size == last) else 0
            if stable >= 3:          # 連續 1.5 秒大小不變
                break
            last = size
            time.sleep(0.5)
        else:
            sys.exit(f"等待 Edge 輸出 PDF 逾時（exit {r.returncode}）：\n{r.stderr[-2000:]}")
        time.sleep(1.0)              # 讓 Edge 子程序釋放檔案與設定檔目錄
    finally:
        shutil.rmtree(prof, ignore_errors=True)
    print(f"寫出 {PDF}（{PDF.stat().st_size / 1024:.0f} KB）")


def render_pages(dpi: int) -> int:
    PAGES.mkdir(parents=True, exist_ok=True)
    for old in PAGES.glob("page_*.png"):
        old.unlink()
    pdf = pdfium.PdfDocument(str(PDF))
    n = len(pdf)
    for i in range(n):
        page = pdf[i]
        img = page.render(scale=dpi / 72).to_pil()
        out = PAGES / f"page_{i + 1}.png"
        img.save(out)
        w_mm, h_mm = (v / 72 * 25.4 for v in page.get_size())
        print(f"寫出 {out}（{img.width}×{img.height} px；頁面 {w_mm:.0f}×{h_mm:.0f} mm）")
    pdf.close()
    return n


class AbstractText(HTMLParser):
    """擷取 <section class="abstract"> 內 <p> 的文字。"""

    def __init__(self):
        super().__init__()
        self.in_abs = self.in_p = False
        self.text = []

    def handle_starttag(self, tag, attrs):
        if tag == "section" and ("class", "abstract") in attrs:
            self.in_abs = True
        elif tag == "p" and self.in_abs:
            self.in_p = True

    def handle_endtag(self, tag):
        if tag == "p":
            self.in_p = False
        elif tag == "section":
            self.in_abs = False

    def handle_data(self, data):
        if self.in_p:
            self.text.append(data)


def check_abstract() -> bool:
    p = AbstractText()
    p.feed(HTML.read_text(encoding="utf-8"))
    s = "".join(p.text)
    n_nospace = len(re.sub(r"\s", "", s))
    n_han = len(re.findall(r"[\u4e00-\u9fff]", s))
    ok = ABS_MIN <= n_nospace <= ABS_MAX
    print(f"摘要字數：{n_nospace}（不含空白；其中漢字 {n_han}）— {'OK' if ok else '超出 100–200'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args()
    print_pdf(find_edge())
    n = render_pages(args.dpi)
    ok_pages = PAGE_MIN <= n <= PAGE_MAX
    print(f"總頁數：{n}（限制 {PAGE_MIN}–{PAGE_MAX}）— {'OK' if ok_pages else '不符規定'}")
    ok_abs = check_abstract()
    if not (ok_pages and ok_abs):
        sys.exit(1)


if __name__ == "__main__":
    main()
