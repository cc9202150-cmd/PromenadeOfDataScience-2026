# -*- coding: utf-8 -*-
"""長版報告轉檔：report_long/chapters/*.md → DOCX（python-docx）→ PDF（Microsoft Word COM）→ 逐頁 PNG。

一份來源產生兩種交件格式：
  1. 依檔名排序讀入章節 Markdown（支援的寫法見下方「Markdown 方言」），
     以 python-docx 組成 DOCX：A4、邊界 2.54 cm、單行間距、內文 12 pt（中文新細明體＋西文 Times New Roman）、
     標題微軟正黑體粗體並自動加上 1／1.1／1.1.1 編號、封面、前置頁（摘要、Abstract、致謝）、目錄（Word TOC 欄位）、
     圖目錄與表目錄、全書統一的圖表編號與交叉參照、頁尾頁碼（前置頁 i、ii…；本文 1、2…）。
  2. 以 PowerShell 啟動 Microsoft Word（COM）：開啟 DOCX → 更新所有欄位與目錄 → 存回 DOCX → 匯出 PDF
     （ExportAsFixedFormat，含標題書籤）→ 無論成功與否都關閉 Word。DOCX 與 PDF 因此版面一致。
  3. 以 pypdfium2 將 PDF 每頁轉成 PNG（預設 90 dpi），供逐頁目視檢查。
  4. 列印格式自我檢查：頁數、頁面尺寸、邊界、各類段落字級、行距規則、目錄是否存在與條目數、PDF 實際字型與字級。

Markdown 方言（章節撰寫者只能用這些）
  # / ## / ### 標題（自動編號，勿手打數字）；#### 不編號小標
  段落以空行分隔；行內 **粗體**、*斜體*、`程式碼`；\\* 表示字面星號（緊接在數字或括號後的 *、**、*** 一律視為顯著性星號）
  - 項目符號清單、1. 編號清單（可再縮排兩格一層）
  :::table id=tab:xxx 表題   → 接著是 pipe table，可再接 :::note 表註
  :::figure id=fig:xxx path=figures/xxx.png width=16 | 圖說
  :::formula 置中的文字公式
  交叉參照 圖{fig:xxx}、表{tab:xxx} → 圖N、表N（id 不存在即報錯）
特殊規則
  * 檔名以 00 開頭者為前置頁：其 # 標題（摘要、Abstract、致謝）不編號、各自另起一頁，列入目錄（頁碼 i、ii…）。
  * 「生成式 AI 使用聲明」「參考文獻」不編號（進目錄）；AI 聲明不換頁、接在結論之後。
    「附錄 X　…」不再編號（字母由撰寫者手打），## 以下編為 X.1。
  * 參考文獻章內的 - 清單項目轉為懸掛縮排 0.75 cm 的條目（去掉項目符號），網址轉為可點擊連結。
  * 不以 # 開頭的檔案（例如 05b_results.md）延續上一章的節編號。檔名以 _ 開頭的 .md 不讀入（草稿）。
  * 表格欄寬依內容自動計算（固定版面）；欄太多放不下時改用 10 pt、縮小內距，並在提醒中列出。
    統計式（p = .005、[0.031, 0.096]、95% CI）以不換行空白綁住；程式碼在 / 與 _ 後、網址在 / 後可斷行。

執行
  .venv/Scripts/python -I scripts/08_build_long_report.py                     # 正式版
  .venv/Scripts/python -I scripts/08_build_long_report.py --src report_long/chapters_sample --out chapters_sample/build/sample
選項
  --src DIR       章節資料夾（預設 report_long/chapters）
  --out NAME      輸出主檔名（不含副檔名；相對於 report_long，或絕對路徑；預設 夜越深心越沉_完整報告）
  --pages DIR     PNG 輸出資料夾（預設：輸出檔所在資料夾下的 pages/）
  --dpi N         PNG 解析度（預設 90）
  --no-pdf        只產生 DOCX（不呼叫 Word；目錄須在 Word 中按 F9 更新）
  --no-lists      不產生圖目錄、表目錄
  --toc-first     目錄放在摘要之前（預設：封面 → 摘要／Abstract／致謝 → 目錄 → 圖目錄 → 表目錄 → 本文）
結束代碼：0＝成功；1＝來源有錯（交叉參照找不到、圖檔不存在等）或格式自我檢查未通過；2＝Word 匯出失敗。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import unicodedata
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Pt, RGBColor, Twips
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
RL = ROOT / "report_long"
DEFAULT_SRC = RL / "chapters"
DEFAULT_OUT = "夜越深心越沉_完整報告"

META = {
    "title": "夜越深，心越沉？",
    "subtitle": "國中生晚睡、上網與一年後的憂鬱",
    "title_en": ("The Later the Night, the Heavier the Heart? Late Bedtimes, Internet Use, and "
                 "Depressive Symptoms One Year Later among Taiwanese Junior High Students"),
    "authors": "林冠穎、羅尹",
    "date": "2026 年 10 月",
    "keywords": "青少年憂鬱; 上床時間; 螢幕使用; 交叉延宕模型; 數位教養",
}
RUNNING_HEAD = "夜越深，心越沉？國中生晚睡、上網與一年後的憂鬱"

# ── 版面與字型 ─────────────────────────────────────────────────────────────
FONT_CJK = "新細明體"          # PMingLiU
FONT_LATIN = "Times New Roman"
FONT_HEAD = "微軟正黑體"       # Microsoft JhengHei
FONT_CODE = "Consolas"
PAGE_W_CM, PAGE_H_CM, MARGIN_CM = 21.0, 29.7, 2.54
TEXT_W_CM = PAGE_W_CM - 2 * MARGIN_CM                    # 15.92 cm
TEXT_W_TW = round(TEXT_W_CM / 2.54 * 1440)                # 9026 twips
BODY_PT, CAPTION_PT, TABLE_PT, NOTE_PT = 12, 10.5, 10.5, 10        # 行內程式碼沿用所在段落的字級（內文 12 pt）
HEAD_PT = {1: 18, 2: 15, 3: 13, 4: 12}
MAX_FIG_H_CM = 20.5
CELL_MAR_TW = 85                                          # 表格儲存格左右內距（0.15 cm）
BLACK = RGBColor(0, 0, 0)
GREY = RGBColor(0x59, 0x59, 0x59)
HEADER_FILL = "D9D9D9"

FRONT_PREFIX = "00"
UNNUMBERED = {"摘要", "Abstract", "致謝", "參考文獻", "References", "生成式AI使用聲明", "目錄"}
NO_PAGEBREAK = {"生成式AI使用聲明"}
REFERENCE_TITLES = {"參考文獻", "References"}
APPENDIX_RE = re.compile(r"^附錄\s*([A-Z])(?![A-Za-z])")
IDEO_SPACE = "　"
LIST_LINE_EM = 32.5            # 圖目錄／表目錄一行可容納的字寬（em），超過才截短

# pPr 子元素的 schema 順序（插入 python-docx 未提供 API 的元素時用）
PPR_ORDER = ["pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl", "numPr",
             "suppressLineNumbers", "pBdr", "shd", "tabs", "suppressAutoHyphens", "kinsoku", "wordWrap",
             "overflowPunct", "topLinePunct", "autoSpaceDE", "autoSpaceDN", "bidi", "adjustRightInd",
             "snapToGrid", "spacing", "ind", "contextualSpacing", "mirrorIndents", "suppressOverlap", "jc",
             "textDirection", "textAlignment", "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr",
             "sectPr", "pPrChange"]
SECTPR_ORDER = ["headerReference", "footerReference", "footnotePr", "endnotePr", "type", "pgSz", "pgMar",
                "paperSrc", "pgBorders", "lnNumType", "pgNumType", "cols", "formProt", "vAlign", "noEndnote",
                "titlePg", "textDirection", "bidi", "rtlGutter", "docGrid", "printerSettings", "sectPrChange"]


# ══════════════════════════════════════════════════════════════════════════
# 1. Markdown 解析
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class Block:
    kind: str                      # heading / para / list / table / figure / formula / note
    loc: str                       # 檔名:行號（錯誤訊息用）
    level: int = 0
    text: str = ""
    items: list = field(default_factory=list)     # list：[(depth, ordered, start, text)]
    rows: list = field(default_factory=list)      # table：第一列為標題列
    aligns: list = field(default_factory=list)    # table：每欄 'l' / 'c' / 'r' / None（自動）
    note: str = ""
    fid: str = ""                  # 圖表 id
    path: str = ""
    width: float = 0.0


@dataclass
class Chapter:
    path: Path
    blocks: list


class Problems:
    def __init__(self):
        self.errors, self.warnings = [], []

    def error(self, loc, msg):
        self.errors.append(f"{loc}  {msg}")

    def warn(self, loc, msg):
        self.warnings.append(f"{loc}  {msg}")


HEAD_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
LIST_RE = re.compile(r"^(?P<ind> *)(?P<mk>[-*+]|\d{1,3}[.)])\s+(?P<text>.*)$")
TABLE_RE = re.compile(r"^:::table\s+id=(?P<id>tab:[\w.\-]+)\s*(?P<cap>.*)$")
FIG_RE = re.compile(r"^:::figure\s+id=(?P<id>fig:[\w.\-]+)\s+path=(?P<path>\S+)"
                    r"(?:\s+width=(?P<w>\d+(?:\.\d+)?)\s*(?:cm)?)?\s*\|\s*(?P<cap>.*)$")
NOTE_RE = re.compile(r"^:::note\b\s*(?P<text>.*)$")
FORMULA_RE = re.compile(r"^:::formula\b\s*(?P<text>.*)$")
DIRECTIVE_RE = re.compile(r"^:::(\w*)")
ALIGN_CELL_RE = re.compile(r"^:?-{2,}:?$")


def is_wide(ch: str) -> bool:
    return unicodedata.east_asian_width(ch) in ("W", "F")


def smart_join(parts: list[str]) -> str:
    """多行合併為一段：中文接中文不加空白，西文之間加一個空白。"""
    out = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if out and not (is_wide(out[-1]) or is_wide(p[0])):
            out += " "
        out += p
    return out


def split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    cells, buf, i = [], "", 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s) and s[i + 1] == "|":
            buf += "|"
            i += 2
            continue
        if s[i] == "|":
            cells.append(buf.strip())
            buf = ""
        else:
            buf += s[i]
        i += 1
    cells.append(buf.strip())
    return cells


def parse_table_lines(lines: list[str], loc: str, prob: Problems):
    rows = [split_row(l) for l in lines]
    aligns = []
    if len(rows) >= 2 and all(ALIGN_CELL_RE.match(c.replace(" ", "")) for c in rows[1] if c != "") and rows[1]:
        for c in rows[1]:
            c = c.replace(" ", "")
            if c.startswith(":") and c.endswith(":"):
                aligns.append("c")
            elif c.endswith(":"):
                aligns.append("r")
            elif c.startswith(":"):
                aligns.append("l")
            else:
                aligns.append(None)
        rows = [rows[0]] + rows[2:]
    else:
        prob.warn(loc, "表格缺少分隔列（|---|---|），第一列仍視為標題列")
    ncol = max(len(r) for r in rows)
    for r in rows:
        if len(r) != ncol:
            prob.warn(loc, f"表格各列欄數不一致（{len(r)} vs {ncol}），已補空白儲存格")
            r.extend([""] * (ncol - len(r)))
    aligns = (aligns + [None] * ncol)[:ncol]
    return rows, aligns


def is_block_start(line: str) -> bool:
    s = line.strip()
    return bool(HEAD_RE.match(line) or s.startswith(":::") or s.startswith("|") or LIST_RE.match(line))


def parse_markdown(path: Path, prob: Problems) -> list[Block]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    blocks: list[Block] = []
    para: list[str] = []
    para_loc = ""
    n = len(lines)
    i = 0

    def flush():
        nonlocal para
        if para:
            blocks.append(Block("para", para_loc, text=smart_join(para)))
            para = []

    while i < n:
        line = lines[i].rstrip()
        s = line.strip()
        loc = f"{path.name}:{i + 1}"
        if s.startswith("<!--"):                       # HTML 註解：略過（警告）
            flush()
            j = i
            while j < n and "-->" not in lines[j]:
                j += 1
            prob.warn(loc, "HTML 註解已略過")
            i = j + 1
            continue
        if not s:
            flush()
            i += 1
            continue
        m = HEAD_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            if level > 4:
                prob.error(loc, f"不支援 {'#' * level} 標題（最多 ####）")
                level = 4
            blocks.append(Block("heading", loc, level=level, text=m.group(2).strip()))
            i += 1
            continue
        if s.startswith(":::"):
            flush()
            mt, mf, mn, mfo = TABLE_RE.match(s), FIG_RE.match(s), NOTE_RE.match(s), FORMULA_RE.match(s)
            if mt:
                j = i + 1
                while j < n and not lines[j].strip():
                    j += 1
                tl = []
                while j < n and lines[j].strip().startswith("|"):
                    tl.append(lines[j])
                    j += 1
                if not tl:
                    prob.error(loc, f"{mt.group('id')}：:::table 之後找不到 pipe table")
                    i = j
                    continue
                rows, aligns = parse_table_lines(tl, loc, prob)
                blk = Block("table", loc, text=mt.group("cap").strip(), rows=rows, aligns=aligns, fid=mt.group("id"))
                if not blk.text:
                    prob.warn(loc, f"{blk.fid} 沒有表題")
                k = j
                while k < n and not lines[k].strip():
                    k += 1
                if k < n and NOTE_RE.match(lines[k].strip()):
                    note = [NOTE_RE.match(lines[k].strip()).group("text")]
                    k += 1
                    while k < n and lines[k].strip() and not is_block_start(lines[k]):
                        note.append(lines[k])
                        k += 1
                    blk.note = smart_join(note)
                    j = k
                blocks.append(blk)
                i = j
                continue
            if mf:
                w = float(mf.group("w")) if mf.group("w") else 16.0
                blocks.append(Block("figure", loc, text=mf.group("cap").strip(), fid=mf.group("id"),
                                    path=mf.group("path"), width=w))
                i += 1
                continue
            if mfo:
                blocks.append(Block("formula", loc, text=mfo.group("text").strip()))
                i += 1
                continue
            if mn:
                prob.warn(loc, ":::note 前面沒有表格，以表註格式單獨輸出")
                blocks.append(Block("note", loc, text=mn.group("text").strip()))
                i += 1
                continue
            if s.startswith(":::figure") or s.startswith(":::table"):
                prob.error(loc, f"圖表指令格式錯誤：{s[:80]}")
            else:
                prob.error(loc, f"不支援的指令：{DIRECTIVE_RE.match(s).group(0)}")
            i += 1
            continue
        if s.startswith("|"):
            flush()
            tl = []
            j = i
            while j < n and lines[j].strip().startswith("|"):
                tl.append(lines[j])
                j += 1
            prob.warn(loc, "表格前沒有 :::table 指令（不編號、無表題）")
            rows, aligns = parse_table_lines(tl, loc, prob)
            blocks.append(Block("table", loc, rows=rows, aligns=aligns))
            i = j
            continue
        if LIST_RE.match(line):
            flush()
            items = []
            j = i
            while j < n:
                lj = lines[j].rstrip()
                mm = LIST_RE.match(lj)
                if mm:
                    depth = 1 if len(mm.group("ind")) >= 2 else 0
                    mk = mm.group("mk")
                    ordered = mk[0].isdigit()
                    start = int(mk[:-1]) if ordered else 1
                    items.append([depth, ordered, start, [mm.group("text")]])
                    j += 1
                    continue
                if not lj.strip():
                    # 空行之後若仍是清單項目，視為同一清單（鬆散清單）
                    k = j
                    while k < n and not lines[k].strip():
                        k += 1
                    if k < n and LIST_RE.match(lines[k].rstrip()):
                        j = k
                        continue
                    break
                if is_block_start(lj):
                    break
                items[-1][3].append(lj)          # 延續行
                j += 1
            blocks.append(Block("list", loc, items=[(d, o, st, smart_join(t)) for d, o, st, t in items]))
            i = j
            continue
        if s in ("---", "***", "___"):
            flush()
            prob.warn(loc, "水平線已略過")
            i += 1
            continue
        if s.startswith(">"):
            prob.warn(loc, "不支援引言區塊（>），以一般段落輸出")
            s = s.lstrip("> ").strip()
        if not para:
            para_loc = loc
        para.append(s)
        i += 1
    flush()
    return blocks


# ══════════════════════════════════════════════════════════════════════════
# 2. 圖表編號與交叉參照
# ══════════════════════════════════════════════════════════════════════════
XREF_RE = re.compile(r"\{(fig|tab):([\w.\-]+)\}")
BAD_XREF_RE = re.compile(r"\{\s*(?:fig|tab)\s*:\s*[^}]*\}|\{(?:fig|tab)[^:}][^}]*\}")
HANDNUM_RE = re.compile(r"(?<![圖表附代量發列報課外儀版年])[圖表]\s?\d{1,2}(?![\d.,]|\s?[校人題項筆])")


def number_floats(chapters: list[Chapter], prob: Problems) -> dict:
    numbers, counters = {}, {"fig": 0, "tab": 0}
    for ch in chapters:
        for b in ch.blocks:
            if b.kind in ("figure", "table") and b.fid:
                kind = b.fid.split(":")[0]
                if b.fid in numbers:
                    prob.error(b.loc, f"重複的 id：{b.fid}")
                    continue
                counters[kind] += 1
                numbers[b.fid] = counters[kind]
                b.level = counters[kind]
            elif b.kind == "table":
                b.level = 0
    return numbers


def resolve_xrefs(chapters: list[Chapter], numbers: dict, prob: Problems) -> Counter:
    used = Counter()

    def sub(text, loc):
        def rep(m):
            key = f"{m.group(1)}:{m.group(2)}"
            if key not in numbers:
                prob.error(loc, f"交叉參照找不到 id：{{{key}}}")
                return m.group(0)
            used[key] += 1
            return str(numbers[key])
        out = XREF_RE.sub(rep, text)
        for bad in BAD_XREF_RE.findall(XREF_RE.sub("", text)):
            prob.error(loc, f"交叉參照格式錯誤：{bad}")
        return out

    for ch in chapters:
        for b in ch.blocks:
            b.text = sub(b.text, b.loc)
            b.note = sub(b.note, b.loc)
            b.items = [(d, o, s, sub(t, b.loc)) for d, o, s, t in b.items]
            b.rows = [[sub(c, b.loc) for c in r] for r in b.rows]
    for key in numbers:
        if not used[key]:
            prob.warn("（全書）", f"{key} 在正文中沒有被參照")
    return used


def check_handnumbers(chapters: list[Chapter], prob: Problems):
    """正文手打的「圖3」「表 2」提醒（交叉參照已替換後無法區分，故在替換前檢查）。"""
    for ch in chapters:
        for b in ch.blocks:
            texts = [b.text] if b.kind in ("para", "heading", "formula") else [t for *_, t in b.items]
            for t in texts:
                for m in HANDNUM_RE.finditer(XREF_RE.sub("", t)):
                    prob.warn(b.loc, f"疑似手打圖表編號「{m.group(0)}」，請改用 圖{{fig:id}}／表{{tab:id}}")


# ══════════════════════════════════════════════════════════════════════════
# 3. 行內格式
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class Seg:
    text: str
    bold: bool = False
    italic: bool = False
    code: bool = False
    url: str = ""


ESCAPABLE = set("\\`*_{}[]()#+-.!|")
URL_RE = re.compile(r"https?://[^\s<>，。；、）」』]+")
EA_AMBIG = set("…—“”‘’·※")


def parse_inline(text: str) -> list[Seg]:
    segs: list[Seg] = []
    buf: list[str] = []
    bold = italic = False
    i, n = 0, len(text)

    def emit():
        if buf:
            segs.append(Seg("".join(buf), bold, italic))
            buf.clear()

    def closing_star(j):
        """從 j 起是否有可作為斜體結尾的單一 *（前一字非空白、不是 ** 的一部分）。"""
        while True:
            j = text.find("*", j)
            if j < 0:
                return False
            if text.startswith("**", j) and not (j + 2 < n and text[j + 2] == "*"):
                j += 2
                continue
            if j > 0 and not text[j - 1].isspace() and text[j - 1] != "\\":
                return True
            j += 1

    while i < n:
        c = text[i]
        if c == "\\" and i + 1 < n and text[i + 1] in ESCAPABLE:
            buf.append(text[i + 1])
            i += 2
            continue
        if c == "`":
            j = text.find("`", i + 1)
            if j > i + 1:
                emit()
                segs.append(Seg(text[i + 1:j], bold, italic, code=True))
                i = j + 1
                continue
        if c == "*" and not bold and not italic and i > 0 and (text[i - 1].isdigit() or text[i - 1] in "]）)"):
            # 顯著性星號（0.049**、[…]*）：緊接在數字或括號之後、且沒有未關閉的強調時，整串 * 視為字面文字
            j = i
            while j < n and text[j] == "*":
                j += 1
            buf.append(text[i:j])
            i = j
            continue
        if text.startswith("**", i):
            # 開頭的 ** 後面不可是空白（CommonMark 左側定界規則），例如圖例「** p < .01」不是粗體
            if not bold and (i + 2 >= n or text[i + 2].isspace()):
                buf.append("**")
                i += 2
                continue
            if bold or text.find("**", i + 2) != -1:
                emit()
                bold = not bold
                i += 2
                continue
        if c == "*":
            if italic and i > 0 and not text[i - 1].isspace():
                emit()
                italic = False
                i += 1
                continue
            if not italic and i + 1 < n and not text[i + 1].isspace() and closing_star(i + 1):
                emit()
                italic = True
                i += 1
                continue
        buf.append(c)
        i += 1
    emit()
    # 網址拆成連結片段
    out: list[Seg] = []
    for s in segs:
        if s.code:
            out.append(s)
            continue
        pos = 0
        for m in URL_RE.finditer(s.text):
            url = m.group(0)
            while url and url[-1] in ".,;:)]":
                if url[-1] == ")" and url.count("(") >= url.count(")"):
                    break
                url = url[:-1]
            if not url:
                continue
            if m.start() > pos:
                out.append(Seg(s.text[pos:m.start()], s.bold, s.italic))
            out.append(Seg(url, s.bold, s.italic, url=url))
            pos = m.start() + len(url)
        if pos < len(s.text):
            out.append(Seg(s.text[pos:], s.bold, s.italic))
    return [s for s in out if s.text]


def plain(text: str) -> str:
    return "".join(s.text for s in parse_inline(text))


NBSP = " "
_TIE_OP_RE = re.compile(r"(?<=\S) ([=<>≤≥≈]) (?=\S)")          # p = .005、n = 3,654、q < .001
_TIE_BRACKET_RE = re.compile(r"\[[^\[\]]{1,40}\]")             # [0.031, 0.096]
_TIE_CI_RE = re.compile(r"(\d+%) (CI)\b")                       # 95% CI


_TIE_ARROW_RE = re.compile(r"(?<=\S) → (?=\S)")                # 12.6% → 28.9%：箭頭留在前一行尾
# 兩端都是短數值的箭頭（.007 → .194、< .001 → .382、2.4% → 12.3%）整組不斷行：
# Word 在中文段落中會把箭頭後的 .194 從中間拆開（.1／94），因此整組綁定，由欄寬規劃預留寬度。
_NUM = r"(?:[<>≤≥][  ])?[−+]?(?:\d[\d,]*)?\.?\d+%?"
_TIE_NUMARROW_RE = re.compile(rf"(?<![^\s（(：:；，、])({_NUM}) → ({_NUM})(?=$|[\s，。；、（(）)：])")


def tie_spaces(text: str, mode: str = "all") -> str:
    """統計式內的空白改為不換行空白，避免「p =」與「.005」、CI 上下限被拆到兩行。
    mode：all＝運算式、箭頭、95% CI、方括號內；ops＝不綁方括號（極窄的表格用）；none＝不處理。"""
    if mode == "none":
        return text
    text = _TIE_OP_RE.sub(lambda m: f"{NBSP}{m.group(1)}{NBSP}", text)
    text = _TIE_ARROW_RE.sub(f"{NBSP}→ ", text)
    text = _TIE_NUMARROW_RE.sub(lambda m: f"{m.group(1).replace(' ', NBSP)}{NBSP}→{NBSP}{m.group(2).replace(' ', NBSP)}",
                                text)
    if mode == "all":
        text = _TIE_BRACKET_RE.sub(lambda m: m.group(0).replace(" ", NBSP), text)
    return _TIE_CI_RE.sub(lambda m: f"{m.group(1)}{NBSP}{m.group(2)}", text)


def split_ea(text: str) -> list[tuple[str, bool]]:
    """把夾在中文之間的「……」「——」「“”」等寬度不明字元獨立成 eastAsia 字型的 run。"""
    out, buf, flag = [], "", False
    for idx, ch in enumerate(text):
        ea = False
        if ch in EA_AMBIG:
            prev = next((c for c in reversed(text[:idx]) if not c.isspace() and c not in EA_AMBIG), "")
            nxt = next((c for c in text[idx + 1:] if not c.isspace() and c not in EA_AMBIG), "")
            ea = bool((prev and is_wide(prev)) or (nxt and is_wide(nxt)))
        if buf and ea != flag:
            out.append((buf, flag))
            buf = ""
        buf += ch
        flag = ea
    if buf:
        out.append((buf, flag))
    return out


# ══════════════════════════════════════════════════════════════════════════
# 4. DOCX 低階工具
# ══════════════════════════════════════════════════════════════════════════
def insert_ordered(parent, child, order):
    """依 schema 順序把 child 插入 parent（同名元素先移除）。"""
    tag = child.tag.split("}")[1]
    for old in parent.findall(qn(f"w:{tag}")):
        parent.remove(old)
    idx = order.index(tag)
    for k, existing in enumerate(list(parent)):
        etag = existing.tag.split("}")[1]
        if etag in order and order.index(etag) > idx:
            existing.addprevious(child)
            return child
    parent.append(child)
    return child


def set_fonts(rpr, latin=FONT_LATIN, east=FONT_CJK, hint=None):
    rf = rpr.get_or_add_rFonts()
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        rf.attrib.pop(qn(a), None)
    if latin:
        for a in ("w:ascii", "w:hAnsi", "w:cs"):
            rf.set(qn(a), latin)
    if east:
        rf.set(qn("w:eastAsia"), east)
    if hint:
        rf.set(qn("w:hint"), hint)


def style_font(style, size=None, bold=None, italic=None, latin=FONT_LATIN, east=FONT_CJK, color=BLACK):
    rpr = style.element.get_or_add_rPr()
    for tag in ("w:color", "w:i", "w:iCs", "w:b", "w:bCs", "w:u"):
        for el in rpr.findall(qn(tag)):
            rpr.remove(el)
    set_fonts(rpr, latin, east)
    f = style.font
    if size is not None:
        f.size = Pt(size)
        szcs = rpr.find(qn("w:szCs"))
        if szcs is None:
            szcs = OxmlElement("w:szCs")
            sz = rpr.find(qn("w:sz"))
            sz.addnext(szcs)
        szcs.set(qn("w:val"), str(int(round(size * 2))))
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic
    if color is not None:
        f.color.rgb = color


def para_fmt(pf, before=0, after=0, align=None, first_chars=None, left=None, hanging=None, right=None,
             keep_next=None, page_break=None, keep_lines=None):
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    if align is not None:
        pf.alignment = align
    if keep_next is not None:
        pf.keep_with_next = keep_next
    if keep_lines is not None:
        pf.keep_together = keep_lines
    if page_break is not None:
        pf.page_break_before = page_break
    if left is not None:
        pf.left_indent = left
    if right is not None:
        pf.right_indent = right
    if hanging is not None:
        pf.first_line_indent = -hanging
    if first_chars is not None:
        pf.first_line_indent = Pt(BODY_PT * first_chars / 100)
        ind = pf._element.get_or_add_pPr().get_or_add_ind()
        ind.set(qn("w:firstLineChars"), str(first_chars))


def set_outline(style_or_par_element_ppr, level):
    el = OxmlElement("w:outlineLvl")
    el.set(qn("w:val"), str(level))
    insert_ordered(style_or_par_element_ppr, el, PPR_ORDER)


def add_field(paragraph, instr: str, placeholder: str = "", size=None):
    def run_with(child):
        r = OxmlElement("w:r")
        if size:
            rpr = OxmlElement("w:rPr")
            sz = OxmlElement("w:sz")
            sz.set(qn("w:val"), str(int(size * 2)))
            rpr.append(sz)
            r.append(rpr)
        r.append(child)
        paragraph._p.append(r)
        return r

    b = OxmlElement("w:fldChar")
    b.set(qn("w:fldCharType"), "begin")
    b.set(qn("w:dirty"), "true")
    run_with(b)
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = f" {instr} "
    run_with(it)
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    run_with(sep)
    if placeholder:
        t = OxmlElement("w:t")
        t.set(qn("xml:space"), "preserve")
        t.text = placeholder
        run_with(t)
    e = OxmlElement("w:fldChar")
    e.set(qn("w:fldCharType"), "end")
    run_with(e)


def add_tc_field(paragraph, entry: str, ident: str, level: int):
    """TC 欄位（不顯示），供圖目錄／表目錄的 TOC \\f 欄位蒐集。"""
    entry = entry.replace('"', "”")
    for typ, child in (("begin", None), (None, f' TC "{entry}" \\f {ident} \\l {level} '), ("end", None)):
        r = OxmlElement("w:r")
        if typ:
            fc = OxmlElement("w:fldChar")
            fc.set(qn("w:fldCharType"), typ)
            r.append(fc)
        else:
            it = OxmlElement("w:instrText")
            it.set(qn("xml:space"), "preserve")
            it.text = child
            r.append(it)
        paragraph._p.append(r)


def add_hyperlink(paragraph, url: str, text: str, bold=False, italic=False):
    r_id = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
    h = OxmlElement("w:hyperlink")
    h.set(qn("r:id"), r_id)
    h.set(qn("w:history"), "1")
    for chunk, ea in split_ea(text):
        r = OxmlElement("w:r")
        rpr = OxmlElement("w:rPr")
        if ea:
            rf = OxmlElement("w:rFonts")
            rf.set(qn("w:hint"), "eastAsia")
            rpr.append(rf)
        if bold:
            rpr.append(OxmlElement("w:b"))
        if italic:
            rpr.append(OxmlElement("w:i"))
        if len(rpr):
            r.append(rpr)
        t = OxmlElement("w:t")
        t.set(qn("xml:space"), "preserve")
        t.text = chunk
        r.append(t)
        h.append(r)
    paragraph._p.append(h)


ZWSP = "​"
_CODE_BREAK_RE = re.compile(r"([/_])(?=[\w.])")


def url_display(url: str) -> str:
    """網址的顯示文字：路徑的每個 / 之後可斷行（零寬空白；連結目標不變），避免長網址把整行撐開。"""
    scheme, sep, rest = url.partition("://")
    if not sep:
        return url
    return scheme + sep + re.sub(r"/(?=[^/])", "/" + ZWSP, rest)


def seg_text(seg: Seg, tie: str = "all") -> str:
    """片段實際輸出的文字：程式碼在 / 與 _ 之後可斷行（零寬空白），一般文字綁定統計式空白；網址在 / 後可斷。"""
    if seg.url:
        return url_display(seg.text)
    if seg.code:
        return _CODE_BREAK_RE.sub(lambda m: m.group(1) + ZWSP, seg.text)
    return tie_spaces(seg.text, tie)


def render_text(text: str, tie: str = "all") -> str:
    return "".join(seg_text(s, tie) for s in parse_inline(text))


def add_inline(paragraph, text: str, bold=False, size=None, tie="all"):
    """把含行內 Markdown 的文字加進段落。bold=True 時全部加粗（例如標題列）；size 覆寫字級（表格縮為 10 pt 時）。"""
    for seg in parse_inline(text):
        if seg.url:
            add_hyperlink(paragraph, seg.url, url_display(seg.text), bold=bold or seg.bold, italic=seg.italic)
            continue
        chunks = [(seg_text(seg, tie), False)] if seg.code else split_ea(seg_text(seg, tie))
        for chunk, ea in chunks:
            run = paragraph.add_run(chunk)
            if bold or seg.bold:
                run.bold = True
            if seg.italic:
                run.italic = True
            if seg.code:
                set_fonts(run._r.get_or_add_rPr(), FONT_CODE, None)
            if size:
                run.font.size = Pt(size)
            if ea:
                set_fonts(run._r.get_or_add_rPr(), None, None, hint="eastAsia")


def sans_bold(paragraph, runs=None):
    """把段落（或指定 run）的中文改用微軟正黑體粗體：新細明體沒有粗體字面，Word 只能假粗體。"""
    for r in (runs if runs is not None else paragraph.runs):
        set_fonts(r._r.get_or_add_rPr(), None, FONT_HEAD)
        r.bold = True


def disp_em(s: str) -> float:
    """以字寬（em）估計文字長度：中文 1、西文約 0.5。"""
    w = 0.0
    for ch in s:
        if is_wide(ch):
            w += 1.0
        elif ch == " ":
            w += 0.25
        elif ch in "mwMW%@—":
            w += 0.85
        elif ch.isupper():
            w += 0.68
        else:
            w += 0.5
    return w


NUMERIC_CELL_RE = re.compile(r"^[\s\d.,%+\-−–—=<>≤≥×≈\[\]()（）、，；／/:;*†pqnNβ]*$")


def is_numeric_cell(s: str) -> bool:
    t = plain(s).strip()
    if not t or t in ("—", "–", "-", "NA", "n/a"):
        return True
    return bool(NUMERIC_CELL_RE.match(t)) and any(ch.isdigit() for ch in t)


CJK_TOKEN_RE = re.compile(r"[⺀-鿿豈-﫿＀-￯　-〿]|[^\s⺀-鿿豈-﫿"
                          r"＀-￯　-〿]+|\s+")


OPEN_PUNCT = set("（「『〔【《〈([{‘“")
CLOSE_PUNCT = set("）」』〕】》〉，。、；：！？)]}%’”…,.;:")


def cell_units(text: str, scale: float = 1.0, tie: str = "all") -> list[tuple[float, bool]]:
    """斷行單位（text 為含行內 Markdown 的儲存格原文，依實際輸出文字計算）：中文逐字、西文逐詞、
    程式碼在零寬空白處可斷；回傳 [(寬度 em, 是否為空白)]。"""
    units: list[list] = []
    glue_next = False
    for tok in CJK_TOKEN_RE.findall(render_text(text, tie).replace(NBSP, "\x00")):   # 不換行空白不是斷點
        if tok.isspace():
            units.append([0.25 * scale, True])
            glue_next = False
            continue
        for k, part in enumerate(tok.split(ZWSP)):
            if not part:
                continue
            w = disp_em(part.replace("\x00", " ")) * scale
            # 避頭尾（kinsoku）：開括號黏後字、標點與閉括號黏前字，Word 不會在這些位置斷行
            if k == 0 and units and not units[-1][1] and (glue_next or part[0] in CLOSE_PUNCT):
                units[-1][0] += w
            else:
                units.append([w, False])
            glue_next = part[-1] in OPEN_PUNCT
    return [(w, sp) for w, sp in units]


def count_lines(units, width: float) -> int:
    lines, cur, pend = 1, 0.0, 0.0
    for w, space in units:
        if space:
            pend = w if cur > 0 else 0.0
            continue
        if cur == 0.0:
            cur = w
        elif cur + pend + w <= width:
            cur += pend + w
        else:
            lines += 1
            cur = w
        pend = 0.0
        while cur > width + 1e-9:          # 單一長詞超過欄寬：Word 會在字元處強制斷行
            lines += 1
            cur -= width
    return lines


@dataclass
class TableLayout:
    widths: list            # 各欄寬（twips）
    pt: float               # 儲存格字級
    margin: int             # 儲存格左右內距（twips）
    tie: str                # 不換行空白的綁定模式（見 tie_spaces）
    feasible: bool          # False＝最小寬加總仍超過版心，只能等比縮小（會被強制斷字）
    height_pt: float        # 估計表格高度（pt）
    label_ok: bool = True   # 第一欄文字標籤是否保有最小寬度


def column_widths(rows: list[list[str]], avail_tw: int, pt: float = TABLE_PT, margin: int = CELL_MAR_TW,
                  tie: str = "all", safety: float = 0.3) -> TableLayout:
    """決定欄寬（固定版面，結果可重現）。
    以 0.25 em 為單位；每欄最小寬＝最長不可斷的西文詞（統計式已用不換行空白綁住），最大寬＝整格不換行。
    目標分數＝Σ（各列行數）² × 100 ＋ 各儲存格行數總和：平方項讓少數特別高的列優先被拉低，表格較勻稱。
    先以貪婪法把剩餘寬度逐次給「每 em 改善最多」的欄，再做兩欄間的寬度移轉局部搜尋，直到無法再改善。"""
    ncol, nrow = len(rows[0]), len(rows)
    pad_em = 2 * margin / (pt * 20)
    avail = avail_tw / (pt * 20) - ncol * pad_em - 0.3
    units = [[cell_units(r[c], 1.06 if ri == 0 else 1.0, tie) for c in range(ncol)]
             for ri, r in enumerate(rows)]
    step = 0.25
    nat_k, mn_k = [], []
    for c in range(ncol):
        col = [units[r][c] for r in range(nrow)]
        nat = max([sum(w for w, _ in u) for u in col] + [1.0]) + 0.35
        longest = max([w for u in col for w, sp in u if not sp] + [0.0])
        mn = max(longest, 2.0) + safety
        mn_k.append(int(-(-mn // step)))
        nat_k.append(max(int(-(-nat // step)), mn_k[-1]))
    total_k = int(avail // step)
    # 第一欄若是文字標籤（類型名稱、構念），最少保留約 5 個中文字寬，避免「早睡/規律/型」式的逐字斷行
    body0 = [r[0] for r in rows[1:] if r[0].strip()]
    label_ok = True
    if body0 and sum(is_numeric_cell(x) for x in body0) / len(body0) < 0.5:
        want = min(nat_k[0], int(-(-5.0 // step)) + int(-(-safety // step)))
        if want > mn_k[0]:
            if sum(mn_k) - mn_k[0] + want < total_k:
                mn_k[0] = want
                nat_k[0] = max(nat_k[0], want)
            else:
                label_ok = False
    cache = {}

    def col_lines(c, k):
        if (c, k) not in cache:
            # 預留 0.25 em 安全邊際：估計字寬與 Word 實際排版略有出入
            cache[(c, k)] = [count_lines(units[r][c], k * step - 0.25) for r in range(nrow)]
        return cache[(c, k)]

    feasible = sum(mn_k) < total_k
    if not feasible:
        w = [k * step * avail / (sum(mn_k) * step) for k in mn_k]
    else:

        def score(ks):
            cols = [col_lines(c, ks[c]) for c in range(ncol)]
            return (sum(max(cols[c][r] for c in range(ncol)) ** 2 for r in range(nrow)) * 100
                    + sum(map(sum, cols)))

        ks = list(mn_k)
        cur = score(ks)
        remaining = total_k - sum(ks)
        while remaining > 0:                                   # 1) 貪婪配置
            best = None
            for c in range(ncol):
                for d in range(1, min(remaining, nat_k[c] - ks[c]) + 1):
                    trial = ks[:c] + [ks[c] + d] + ks[c + 1:]
                    sc = score(trial)
                    if sc < cur:
                        gain = (cur - sc) / d
                        if best is None or gain > best[0] + 1e-9:
                            best = (gain, trial, sc, d)
                        break
            if best is None:
                break
            _, ks, cur, d = best
            remaining -= d
        for _ in range(60):                                    # 2) 兩欄間移轉寬度的局部搜尋
            best = None
            for i in range(ncol):
                for j in range(ncol):
                    if i == j or ks[j] >= nat_k[j]:
                        continue
                    for d in range(1, min(ks[i] - mn_k[i], nat_k[j] - ks[j] + remaining) + 1):
                        trial = list(ks)
                        trial[i] -= d
                        trial[j] += d
                        sc = score(trial)
                        if sc < cur and (best is None or sc < best[1]):
                            best = (trial, sc)
            if best is None:
                break
            ks, cur = best
        w = [k * step for k in ks]
        if remaining > 0:                                      # 3) 剩餘寬度依欄寬比例分配（只會減少換行）
            tot = sum(w)
            w = [x + remaining * step * x / tot for x in w]
        leftover = avail - sum(w)
        w = [x + leftover * x / sum(w) for x in w]
    tw = [int(round((x + pad_em) * pt * 20)) for x in w]
    tw[-1] += avail_tw - sum(tw)
    # 高度估計：每行約 1.3 × 字級，每列另加上下內距與框線約 3 pt
    row_lines = [max(count_lines(units[r][c], w[c] - 0.25) for c in range(ncol)) for r in range(nrow)]
    height = sum(n * pt * 1.3 + 3 for n in row_lines)
    return TableLayout(tw, pt, margin, tie, feasible, height, label_ok)


def plan_table(rows: list[list[str]]) -> TableLayout:
    """依序嘗試：10.5 pt → 10 pt → 縮小內距 → 方括號內可斷行 → 完全不綁定；取第一個放得下的設定。
    欄數 ≥ 8 的表格直接從 10 pt 開始（規格允許表格 10–10.5 pt）。"""
    configs = [(TABLE_PT, CELL_MAR_TW, "all", 0.3), (10, CELL_MAR_TW, "all", 0.3), (10, 57, "all", 0.3),
               (10, 57, "ops", 0.3), (10, 57, "none", 0.3), (10, 40, "none", 0.1)]
    if len(rows[0]) >= 8:
        configs = configs[1:]
    lays = [column_widths(rows, TEXT_W_TW, pt, mar, tie, safety) for pt, mar, tie, safety in configs]
    for lay in lays:                       # 優先：放得下且第一欄標籤保有最小寬度
        if lay.feasible and lay.label_ok:
            return lay
    for lay in lays:
        if lay.feasible:
            return lay
    return lays[-1]


# ══════════════════════════════════════════════════════════════════════════
# 5. 文件建構
# ══════════════════════════════════════════════════════════════════════════
class Builder:
    def __init__(self, prob: Problems, lists: bool, toc_first: bool):
        self.prob = prob
        self.with_lists = lists
        self.toc_first = toc_first
        self.doc = Document()
        self.counts = Counter()
        self.num_seq = 300
        self._setup_document()
        self._setup_styles()

    # ── 文件層級設定 ───────────────────────────────────────────────────
    def _setup_document(self):
        d = self.doc
        sec = d.sections[0]
        sec.page_width, sec.page_height = Cm(PAGE_W_CM), Cm(PAGE_H_CM)
        sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Cm(MARGIN_CM)
        sec.header_distance = sec.footer_distance = Cm(1.27)
        cp = d.core_properties
        cp.created = cp.modified = datetime.now(timezone.utc).replace(microsecond=0)
        cp.revision = 1
        cp.title = f"{META['title']}{META['subtitle']}"
        cp.subject = META["title_en"]
        cp.author = META["authors"]
        cp.last_modified_by = META["authors"]
        cp.keywords = META["keywords"]
        cp.comments = ""
        cp.language = "zh-TW"
        settings = d.settings.element
        # compat：Word 2013+ 版面；主題字型語言改為繁中
        for cs in settings.iter(qn("w:compatSetting")):
            if cs.get(qn("w:name")) == "compatibilityMode":
                cs.set(qn("w:val"), "15")
        tfl = settings.find(qn("w:themeFontLang"))
        if tfl is not None:
            tfl.set(qn("w:eastAsia"), "zh-TW")
        # 佈景主題字型改為本報告字型，任何引用主題字型的文字（例如 Word 產生的目錄 run）都不會退回 Calibri／Cambria
        for rel in d.part.rels.values():
            if rel.reltype.endswith("/theme") and not rel.is_external:
                part = rel.target_part
                xml = part.blob.decode("utf-8")
                for tag, latin, east in (("majorFont", FONT_HEAD, FONT_HEAD), ("minorFont", FONT_LATIN, FONT_CJK)):
                    xml = re.sub(rf"(<a:{tag}>\s*)<a:latin typeface=\"[^\"]*\"/>\s*<a:ea typeface=\"[^\"]*\"/>",
                                 rf'\g<1><a:latin typeface="{latin}"/><a:ea typeface="{east}"/>', xml, count=1)
                    xml = re.sub(rf"(<a:{tag}>.*?<a:font script=\"Hant\" typeface=\")[^\"]*\"",
                                 rf'\g<1>{east}"', xml, count=1, flags=re.S)
                part._blob = xml.encode("utf-8")
        # 文件預設：字型、12 pt、單行間距、段後 0
        dd = d.styles.element.find(qn("w:docDefaults"))
        rpr = dd.find(qn("w:rPrDefault")).find(qn("w:rPr"))
        rf = rpr.find(qn("w:rFonts"))
        for a in list(rf.attrib):
            del rf.attrib[a]
        for a in ("w:ascii", "w:hAnsi", "w:cs"):
            rf.set(qn(a), FONT_LATIN)
        rf.set(qn("w:eastAsia"), FONT_CJK)
        rpr.find(qn("w:sz")).set(qn("w:val"), str(BODY_PT * 2))
        rpr.find(qn("w:szCs")).set(qn("w:val"), str(BODY_PT * 2))
        lang = rpr.find(qn("w:lang"))
        lang.set(qn("w:val"), "en-US")
        lang.set(qn("w:eastAsia"), "zh-TW")
        sp = dd.find(qn("w:pPrDefault")).find(qn("w:pPr")).find(qn("w:spacing"))
        sp.set(qn("w:after"), "0")
        sp.set(qn("w:line"), "240")
        sp.set(qn("w:lineRule"), "auto")

    def _style(self, name, sid=None, base="Normal", next_=None, builtin=False):
        styles = self.doc.styles
        try:
            st = styles[name]
        except KeyError:
            st = styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH, builtin=builtin)
            if sid:
                st.style_id = sid
        if base:
            st.base_style = styles[base]
        if next_:
            st.next_paragraph_style = styles[next_]
        st.quick_style = True
        return st

    def _setup_styles(self):
        S = self.doc.styles
        normal = S["Normal"]
        style_font(normal, BODY_PT)
        para_fmt(normal.paragraph_format, 0, 0)
        normal.paragraph_format.widow_control = True

        body = self._style("Body Text", next_="Body Text")
        style_font(body, BODY_PT)
        para_fmt(body.paragraph_format, 0, 6, WD_ALIGN_PARAGRAPH.JUSTIFY, first_chars=200)
        self.s_body = body

        spacing = {1: (0, 18), 2: (18, 8), 3: (12, 6)}
        for lvl in (1, 2, 3, 4):
            h = S[f"Heading {lvl}"]
            style_font(h, HEAD_PT[lvl], bold=True, italic=False, latin=FONT_HEAD, east=FONT_HEAD)
            b, a = spacing.get(lvl, (10, 4))
            para_fmt(h.paragraph_format, b, a, WD_ALIGN_PARAGRAPH.LEFT, keep_next=True, keep_lines=True,
                     page_break=(lvl == 1))
            h.next_paragraph_style = body
            h.paragraph_format.first_line_indent = Pt(0)
        h1 = S["Heading 1"]
        # 一級標題下方細線
        ppr = h1.element.get_or_add_pPr()
        bdr = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="6" w:space="6" '
                        f'w:color="404040"/></w:pBdr>')
        insert_ordered(ppr, bdr, PPR_ORDER)

        fh = self._style("前置標題", "FrontHeading", next_="Body Text")
        style_font(fh, HEAD_PT[1], bold=True, latin=FONT_HEAD, east=FONT_HEAD)
        para_fmt(fh.paragraph_format, 0, 24, WD_ALIGN_PARAGRAPH.CENTER, keep_next=True, page_break=True)
        set_outline(fh.element.get_or_add_pPr(), 0)   # 摘要、Abstract、致謝：進目錄與 PDF 書籤
        self.s_front = fh
        tt_ = self._style("報告目錄標題", "TocTitle", base="前置標題", next_="Normal")
        set_outline(tt_.element.get_or_add_pPr(), 9)  # 目錄／圖目錄／表目錄標題本身不進目錄
        self.s_toctitle = tt_

        lst = self._style("報告清單", "ListPara", next_="Body Text")
        style_font(lst, BODY_PT)
        para_fmt(lst.paragraph_format, 0, 3, WD_ALIGN_PARAGRAPH.JUSTIFY)
        self.s_list = lst

        fig = self._style("圖", "Figure")
        para_fmt(fig.paragraph_format, 9, 3, WD_ALIGN_PARAGRAPH.CENTER, keep_next=True)
        self.s_fig = fig

        fc = self._style("圖說", "FigureCaption", next_="Body Text")
        style_font(fc, CAPTION_PT)
        para_fmt(fc.paragraph_format, 3, 14, WD_ALIGN_PARAGRAPH.LEFT, keep_lines=True)
        self.s_figcap = fc

        tc = self._style("表題", "TableCaption", next_="Body Text")
        style_font(tc, CAPTION_PT)
        para_fmt(tc.paragraph_format, 9, 4, WD_ALIGN_PARAGRAPH.LEFT, keep_next=True, keep_lines=True)
        self.s_tabcap = tc

        tt = self._style("表格內文", "TableText")
        style_font(tt, TABLE_PT)
        para_fmt(tt.paragraph_format, 0, 0, WD_ALIGN_PARAGRAPH.LEFT)
        self.s_tabtxt = tt

        tn = self._style("表註", "TableNote", next_="Body Text")
        style_font(tn, NOTE_PT)
        para_fmt(tn.paragraph_format, 3, 14, WD_ALIGN_PARAGRAPH.LEFT, keep_lines=True)   # 表註不拆到兩頁
        self.s_note = tn

        ref = self._style("參考文獻條目", "Reference", next_="參考文獻條目")
        style_font(ref, BODY_PT)
        para_fmt(ref.paragraph_format, 0, 6, WD_ALIGN_PARAGRAPH.LEFT, left=Cm(0.75), hanging=Cm(0.75))
        self.s_ref = ref

        fo = self._style("公式", "Formula", next_="Body Text")
        style_font(fo, BODY_PT)
        para_fmt(fo.paragraph_format, 3, 6, WD_ALIGN_PARAGRAPH.CENTER, keep_lines=True)
        self.s_formula = fo

        for name in ("Header", "Footer"):
            st = S[name]
            style_font(st, 9 if name == "Header" else 10.5, color=GREY if name == "Header" else BLACK)
            para_fmt(st.paragraph_format, 0, 0)
            ppr = st.element.get_or_add_pPr()
            for t in ppr.findall(qn("w:tabs")):
                ppr.remove(t)

        # 目錄樣式（Word 內建名稱 toc 1–3；toc 9 給圖目錄、表目錄）
        tabs = (f'<w:tabs {nsdecls("w")}><w:tab w:val="right" w:leader="dot" w:pos="{TEXT_W_TW}"/></w:tabs>')
        toc_spec = {1: (0, 4, 1, True), 2: (480, 0, 1, False), 3: (960, 0, 0, False), 9: (0, 0, 2, False)}
        for lvl, (left, before, after, bold) in toc_spec.items():
            st = self._style(f"toc {lvl}", f"TOC{lvl}", next_="Normal", builtin=True)
            if bold:   # 一級條目與章標題一致：微軟正黑體粗體
                style_font(st, BODY_PT, bold=True, latin=FONT_HEAD, east=FONT_HEAD)
            else:
                style_font(st, BODY_PT, bold=False)
            para_fmt(st.paragraph_format, before, after, WD_ALIGN_PARAGRAPH.LEFT)
            ppr = st.element.get_or_add_pPr()
            insert_ordered(ppr, parse_xml(tabs), PPR_ORDER)
            if lvl == 9:
                st.paragraph_format.left_indent = Cm(1.6)
                st.paragraph_format.first_line_indent = Cm(-1.6)
            else:
                st.paragraph_format.left_indent = Twips(left)
            st.paragraph_format.right_indent = Cm(0.6)

    # ── 段落產生 ───────────────────────────────────────────────────────
    def par(self, style=None, text=None, **kw):
        p = self.doc.add_paragraph(style=style)
        if text:
            add_inline(p, text, **kw)
        return p

    def cover(self):
        cov = self._style("封面", "Cover")
        para_fmt(cov.paragraph_format, 0, 0, WD_ALIGN_PARAGRAPH.CENTER)

        def line(text, size, font=FONT_HEAD, bold=False, before=0, after=0, latin=None, italic=False):
            p = self.doc.add_paragraph(style=cov)
            para_fmt(p.paragraph_format, before, after, WD_ALIGN_PARAGRAPH.CENTER)
            r = p.add_run(text)
            r.font.size = Pt(size)
            r.bold = bold
            r.italic = italic
            set_fonts(r._r.get_or_add_rPr(), latin or font, font)
            return p

        line(META["title"], 28, bold=True, before=150, after=10)
        line(META["subtitle"], 18, before=0, after=20)
        rule = self.doc.add_paragraph(style=cov)
        para_fmt(rule.paragraph_format, 0, 22, left=Cm(5.0), right=Cm(5.0))
        insert_ordered(rule._p.get_or_add_pPr(),
                       parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="8" w:space="1" '
                                 f'w:color="404040"/></w:pBdr>'), PPR_ORDER)
        p = line(META["title_en"], 14, font=FONT_CJK, latin=FONT_LATIN, before=0, after=0)
        p.paragraph_format.left_indent = p.paragraph_format.right_indent = Cm(1.0)
        line(META["authors"], 16, before=210, after=10)
        line(META["date"], 14, font=FONT_CJK, latin=FONT_LATIN, before=6, after=0)

    def section_break(self, page_fmt: str | None, header: bool):
        """在目前位置結束一節；新節的頁碼格式 page_fmt（None＝無頁碼）。"""
        sec = self.doc.add_section(WD_SECTION.NEW_PAGE)
        sectpr = sec._sectPr
        if page_fmt:
            pg = OxmlElement("w:pgNumType")
            pg.set(qn("w:fmt"), page_fmt)
            pg.set(qn("w:start"), "1")
            insert_ordered(sectpr, pg, SECTPR_ORDER)
        sec.footer.is_linked_to_previous = False
        ft = sec.footer.paragraphs[0]
        ft.style = self.doc.styles["Footer"]
        ft.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if page_fmt:
            add_field(ft, "PAGE", "1")
        sec.header.is_linked_to_previous = False
        hd = sec.header.paragraphs[0]
        hd.style = self.doc.styles["Header"]
        hd.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        if header:
            hd.add_run(RUNNING_HEAD)
            insert_ordered(hd._p.get_or_add_pPr(),
                           parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="4" w:space="3" '
                                     f'w:color="A6A6A6"/></w:pBdr>'), PPR_ORDER)
        return sec

    def toc_pages(self, has_figs: bool, has_tabs: bool):
        p = self.par(self.s_toctitle)
        p.add_run(f"目{IDEO_SPACE}錄")
        p = self.doc.add_paragraph(style="Normal")
        add_field(p, 'TOC \\o "1-3" \\h \\z', "（目錄將由 Microsoft Word 更新；在 Word 中可按 F9 更新）")
        if not self.with_lists:
            return
        for ident, title, ok in (("F", f"圖目錄", has_figs), ("T", f"表目錄", has_tabs)):
            if not ok:
                continue
            p = self.par(self.s_toctitle)
            p.add_run(title)
            p = self.doc.add_paragraph(style="Normal")
            add_field(p, f'TOC \\h \\z \\f {ident} \\l "9-9"', f"（{title}將由 Microsoft Word 更新）")

    # ── 區塊 ───────────────────────────────────────────────────────────
    def heading(self, level: int, number: str, text: str, mode: str, page_break: bool | None = None,
                space_before: float | None = None):
        if mode == "front":
            style = self.s_front if level == 1 else self.doc.styles["Heading 4"]
        else:
            style = self.doc.styles[f"Heading {min(level, 4)}"]
        p = self.doc.add_paragraph(style=style)
        if number:
            p.add_run(number + IDEO_SPACE)
        add_inline(p, text)
        if page_break is not None:
            p.paragraph_format.page_break_before = page_break
        if space_before is not None:
            p.paragraph_format.space_before = Pt(space_before)
        self.counts[f"h{level}"] += 1
        return p

    def paragraph(self, text: str):
        p = self.par(self.s_body, text)
        t = text.lstrip()
        # 含長網址的段落改為靠左：網址不能在中間斷開，左右對齊會把前一行的中文字距撐開
        if any(len(m.group(0)) >= 30 for m in URL_RE.finditer(plain(text))):
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            if URL_RE.fullmatch(plain(text).strip()):
                p.paragraph_format.first_line_indent = Pt(0)
                ind = p._p.get_or_add_pPr().get_or_add_ind()
                ind.set(qn("w:firstLineChars"), "0")
        if re.match(r"^\*\*(關鍵詞|關鍵字|Keywords?)", t, re.I):
            p.paragraph_format.first_line_indent = Pt(0)
            ind = p._p.get_or_add_pPr().get_or_add_ind()
            ind.set(qn("w:firstLineChars"), "0")
            p.paragraph_format.space_before = Pt(12)
        self.counts["para"] += 1
        return p

    def _new_num(self, fmt0: str, fmt1: str, start: int) -> int:
        numbering = self.doc.part.numbering_part.element
        self.num_seq += 1
        aid = nid = self.num_seq

        def lvl(il, fmt, st):
            if fmt == "bullet":
                text, left, hang, f = ("•" if il == 0 else "–"), (840 if il == 0 else 1320), 360, "bullet"
            else:
                text, left, hang, f = (f"%{il + 1}." if il == 0 else f"(%{il + 1})"), \
                                      (900 if il == 0 else 1440), (420 if il == 0 else 540), "decimal"
            return (f'<w:lvl w:ilvl="{il}"><w:start w:val="{st}"/><w:numFmt w:val="{f}"/>'
                    f'<w:lvlText w:val="{text}"/><w:lvlJc w:val="left"/>'
                    f'<w:pPr><w:ind w:left="{left}" w:hanging="{hang}"/></w:pPr>'
                    f'<w:rPr><w:rFonts w:ascii="{FONT_LATIN}" w:hAnsi="{FONT_LATIN}" w:eastAsia="{FONT_CJK}" '
                    f'w:cs="{FONT_LATIN}"/></w:rPr></w:lvl>')

        absn = parse_xml(f'<w:abstractNum {nsdecls("w")} w:abstractNumId="{aid}">'
                         f'<w:multiLevelType w:val="hybridMultilevel"/>'
                         f'{lvl(0, fmt0, start)}{lvl(1, fmt1, 1)}</w:abstractNum>')
        first_num = numbering.find(qn("w:num"))
        if first_num is not None:
            first_num.addprevious(absn)
        else:
            numbering.append(absn)
        numbering.append(parse_xml(f'<w:num {nsdecls("w")} w:numId="{nid}"><w:abstractNumId w:val="{aid}"/></w:num>'))
        return nid

    def list_block(self, items):
        fmt0 = next(("decimal" if o else "bullet" for d, o, s, t in items if d == 0), "bullet")
        fmt1 = next(("decimal" if o else "bullet" for d, o, s, t in items if d == 1), "bullet")
        start = next((s for d, o, s, t in items if d == 0), 1)
        nid = self._new_num(fmt0, fmt1, start)
        for depth, ordered, st, text in items:
            p = self.par(self.s_list, text)
            numpr = p._p.get_or_add_pPr().get_or_add_numPr()
            numpr.get_or_add_ilvl().val = depth
            numpr.get_or_add_numId().val = nid
        last = self.doc.paragraphs[-1]
        last.paragraph_format.space_after = Pt(6)
        self.counts["list"] += 1

    def reference(self, text: str):
        self.par(self.s_ref, text)
        self.counts["ref"] += 1

    def formula(self, text: str):
        self.par(self.s_formula, text)
        self.counts["formula"] += 1

    def note(self, text: str):
        t = text.strip()
        if not re.match(r"^(\*{0,2})(註|資料|說明|Note)", t):
            t = "註：" + t
        self.par(self.s_note, t)

    def _caption_layout(self, p, label: str, text: str, box_cm: float):
        width_em = box_cm / 2.54 * 72 / CAPTION_PT
        if disp_em(label + IDEO_SPACE + plain(text)) <= width_em - 0.5:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        else:
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT      # 多行圖說靠左，避免西文詞間被撐開

    @staticmethod
    def short_title(caption: str) -> str:
        t = plain(caption).strip()
        depth = 0
        for k, ch in enumerate(t):
            if ch in "（(":
                depth += 1
            elif ch in "）)":
                depth = max(depth - 1, 0)
            elif ch in "；。" and depth == 0 and k > 4:
                t = t[:k]
                break
        if disp_em(t) > LIST_LINE_EM:         # 仍超過一行：去掉結尾的括號補充說明
            m = re.match(r"^(.{10,}?)\s*[（(][^（）()]*[）)]?$", t)
            if m:
                t = m.group(1).rstrip("：:，, ")
        if disp_em(t) > LIST_LINE_EM:         # 再截在第一個頂層冒號（或夠長的逗號前段）
            depth = 0
            for k, ch in enumerate(t):
                if ch in "（(":
                    depth += 1
                elif ch in "）)":
                    depth = max(depth - 1, 0)
                elif depth == 0 and ((ch in "：:" and disp_em(t[:k]) >= 6)
                                     or (ch in "，," and disp_em(t[:k]) >= 12)):
                    t = t[:k]
                    break
        return t if len(t) <= 70 else t[:68] + "…"

    def figure(self, b: Block):
        path = (RL / b.path).resolve()
        if not path.is_file():
            self.prob.error(b.loc, f"{b.fid}：找不到圖檔 {b.path}（相對於 report_long）")
            return
        width = b.width
        if width > TEXT_W_CM:
            if width > 16.05:
                self.prob.warn(b.loc, f"{b.fid}：寬度 {width} cm 超過版心，改為 {TEXT_W_CM:.2f} cm")
            width = TEXT_W_CM                     # width=16 視為滿版（版心 15.92 cm）
        with Image.open(path) as im:
            wpx, hpx = im.size
        if width * hpx / wpx > MAX_FIG_H_CM:
            width = MAX_FIG_H_CM * wpx / hpx
            self.prob.warn(b.loc, f"{b.fid}：圖太高，寬度縮為 {width:.1f} cm")
        p = self.par(self.s_fig)
        shape = p.add_run().add_picture(str(path), width=Cm(width))
        docpr = shape._inline.find(qn("wp:docPr"))
        if docpr is not None:
            docpr.set("descr", f"圖{b.level}　{plain(b.text)}")
        cap = self.par(self.s_figcap)
        box = max(width, 12.0)
        if box < TEXT_W_CM - 0.2:
            side = Cm((TEXT_W_CM - box) / 2)
            cap.paragraph_format.left_indent = cap.paragraph_format.right_indent = side
        label = f"圖{b.level}"
        sans_bold(cap, [cap.add_run(label + IDEO_SPACE)])
        add_inline(cap, b.text)
        self._caption_layout(cap, label, b.text, box)
        if self.with_lists:
            add_tc_field(cap, f"{label}{IDEO_SPACE}{self.short_title(b.text)}", "F", 9)
        self.counts["figure"] += 1

    def table(self, b: Block):
        rows, aligns = b.rows, list(b.aligns)
        ncol = len(rows[0])
        if b.fid:
            cap = self.par(self.s_tabcap)
            label = f"表{b.level}"
            sans_bold(cap, [cap.add_run(label + IDEO_SPACE)])
            add_inline(cap, b.text)
            self._caption_layout(cap, label, b.text, TEXT_W_CM)
            if self.with_lists:
                add_tc_field(cap, f"{label}{IDEO_SPACE}{self.short_title(b.text)}", "T", 9)
        # 自動對齊：數字欄置中，其餘靠左
        for c in range(ncol):
            if aligns[c] is None:
                body = [r[c] for r in rows[1:] if r[c].strip()]
                numeric = body and sum(is_numeric_cell(x) for x in body) / len(body) >= 0.7
                aligns[c] = "c" if numeric else "l"
        lay = plan_table(rows)
        widths, size = lay.widths, (lay.pt if lay.pt != TABLE_PT else None)
        if not lay.feasible:
            self.prob.warn(b.loc, f"{b.fid or '表格'}：{ncol} 欄太寬，即使 10 pt 仍放不下，部分儲存格會被強制斷字；"
                                  f"建議拆表或減少欄數")
        elif lay.pt != TABLE_PT:
            self.counts["table_10pt"] += 1
        tbl = self.doc.add_table(rows=len(rows), cols=ncol)
        t = tbl._tbl
        tblpr = t.tblPr
        for child in list(tblpr):
            if child.tag != qn("w:tblStyle"):
                tblpr.remove(child)
        tblpr.append(parse_xml(f'<w:tblW {nsdecls("w")} w:w="{sum(widths)}" w:type="dxa"/>'))
        tblpr.append(parse_xml(f'<w:jc {nsdecls("w")} w:val="center"/>'))
        bd = "".join(f'<w:{side} w:val="single" w:sz="{sz}" w:space="0" w:color="000000"/>'
                     for side, sz in (("top", 8), ("left", 4), ("bottom", 8), ("right", 4),
                                      ("insideH", 4), ("insideV", 4)))
        tblpr.append(parse_xml(f'<w:tblBorders {nsdecls("w")}>{bd}</w:tblBorders>'))
        tblpr.append(parse_xml(f'<w:tblLayout {nsdecls("w")} w:type="fixed"/>'))
        tblpr.append(parse_xml(f'<w:tblCellMar {nsdecls("w")}><w:top w:w="28" w:type="dxa"/>'
                               f'<w:left w:w="{lay.margin}" w:type="dxa"/><w:bottom w:w="28" w:type="dxa"/>'
                               f'<w:right w:w="{lay.margin}" w:type="dxa"/></w:tblCellMar>'))
        tblpr.append(parse_xml(f'<w:tblLook {nsdecls("w")} w:val="04A0" w:firstRow="1" w:lastRow="0" '
                               f'w:firstColumn="0" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/>'))
        grid = t.tblGrid
        for gc, w in zip(grid.findall(qn("w:gridCol")), widths):
            gc.set(qn("w:w"), str(w))
        # 表頭列在跨頁時重複；表題一律與表頭同頁（以 keep_with_next 串起），不會單獨留在前一頁。
        note_pt = (disp_em(plain(b.note)) / (TEXT_W_CM / 2.54 * 72 / NOTE_PT) + 1) * NOTE_PT * 1.3 if b.note else 0
        body_h = (PAGE_H_CM - 2 * MARGIN_CM) / 2.54 * 72
        # 整張不跨頁：估計高度（含表題、表註）不超過版心 30%。其餘允許跨頁，但表頭＋前 3 列
        # 與表題同頁、最後 3 列與表註同頁、分組標題列（只有第一欄有字）與下一列同頁，避免整張表被推到下一頁而
        # 在前一頁留下大片空白，也避免跨頁後只剩一兩列。
        keep = lay.height_pt + 24 + note_pt <= 0.30 * body_h
        head_n, tail_n = 3, 3
        amap = {"l": WD_ALIGN_PARAGRAPH.LEFT, "c": WD_ALIGN_PARAGRAPH.CENTER, "r": WD_ALIGN_PARAGRAPH.RIGHT}
        for ri, (row, xrow) in enumerate(zip(rows, tbl.rows)):
            trpr = xrow._tr.get_or_add_trPr()
            trpr.append(OxmlElement("w:cantSplit"))
            if ri == 0:
                trpr.append(OxmlElement("w:tblHeader"))
            for ci, (txt, cell) in enumerate(zip(row, xrow.cells)):
                cell.width = Twips(widths[ci])
                if ri == 0:
                    tcpr = cell._tc.get_or_add_tcPr()
                    tcpr.append(parse_xml(f'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="{HEADER_FILL}"/>'))
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                p = cell.paragraphs[0]
                p.style = self.s_tabtxt
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if ri == 0 else amap[aligns[ci]]
                add_inline(p, txt, bold=(ri == 0), size=size, tie=lay.tie)
                if ri == 0:
                    sans_bold(p)
                last = ri == len(rows) - 1
                group_row = ri > 0 and not any(plain(c).strip() for c in row[1:])
                if (keep and (not last or b.note)) or (not keep and (ri < head_n or group_row
                                                                     or (ri >= len(rows) - tail_n and (not last or b.note)))):
                    p.paragraph_format.keep_with_next = True       # 長表：最後一列與表註同頁，表註不會單獨落到下一頁
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        if b.note:
            self.note(b.note)
        else:
            # 表後留白（無表註時）：6 pt 的空段落隔開下一段
            sp = self.doc.add_paragraph(style="Normal")
            sp.paragraph_format.space_after = Pt(6)
            insert_ordered(sp._p.get_or_add_pPr(),
                           parse_xml(f'<w:rPr {nsdecls("w")}><w:sz w:val="12"/><w:szCs w:val="12"/></w:rPr>'),
                           PPR_ORDER)
        self.counts["table"] += 1

    # ── 主流程 ─────────────────────────────────────────────────────────
    def build(self, chapters: list[Chapter]):
        front = [c for c in chapters if c.path.name.startswith(FRONT_PREFIX)]
        body = [c for c in chapters if not c.path.name.startswith(FRONT_PREFIX)]
        has_figs = any(b.kind == "figure" for c in chapters for b in c.blocks)
        has_tabs = any(b.kind == "table" and b.fid for c in chapters for b in c.blocks)

        self.cover()
        self.section_break("lowerRoman", header=False)          # 前置頁：i, ii, …
        first_front = True
        if self.toc_first:
            self.toc_pages(has_figs, has_tabs)
            first_front = False
        for ch in front:
            for b in ch.blocks:
                if b.kind == "heading":
                    p = self.heading(b.level, "", b.text, "front")
                    if b.level == 1 and first_front:
                        p.paragraph_format.page_break_before = False
                        first_front = False
                else:
                    self.emit_block(b, refs=False)
        if not self.toc_first:
            self.toc_pages(has_figs, has_tabs)
        self.section_break("decimal", header=True)              # 本文：1, 2, …

        c1 = c2 = c3 = 0
        mode, letter = "body", ""
        refs = False
        first_h1 = True
        for ch in body:
            for b in ch.blocks:
                if b.kind != "heading":
                    self.emit_block(b, refs=refs)
                    continue
                key = re.sub(r"\s+", "", plain(b.text))
                if b.level == 1:
                    am = APPENDIX_RE.match(plain(b.text))
                    refs = key in REFERENCE_TITLES
                    if am:
                        mode, letter, c2, c3 = "appendix", am.group(1), 0, 0
                        num = ""
                    elif key in UNNUMBERED:
                        mode, num = "unnumbered", ""
                    else:
                        mode, num = "body", None
                        c1, c2, c3 = c1 + 1, 0, 0
                        num = str(c1)
                    pb = False if (first_h1 or key in NO_PAGEBREAK) else None
                    # 不換頁的一級標題（生成式 AI 使用聲明）接在前一章之後，上方留白
                    sb = 30 if (key in NO_PAGEBREAK and not first_h1) else None
                    first_h1 = False
                    self.heading(1, num, b.text, mode, page_break=pb, space_before=sb)
                elif b.level in (2, 3):
                    if b.level == 2:
                        c2, c3 = c2 + 1, 0
                    else:
                        c3 += 1
                    prefix = str(c1) if mode == "body" else (letter if mode == "appendix" else "")
                    if not prefix:
                        num = ""
                    elif b.level == 2:
                        num = f"{prefix}.{c2}"
                    else:
                        num = f"{prefix}.{c2}.{c3}"
                    if mode == "body" and c1 == 0:
                        self.prob.warn(b.loc, "第一個 # 章標題之前出現 ## / ###，編號以 0 起算")
                    self.heading(b.level, num, b.text, mode)
                else:
                    self.heading(4, "", b.text, mode)
        return self.doc

    def emit_block(self, b: Block, refs: bool):
        if b.kind == "para":
            self.paragraph(b.text)
        elif b.kind == "list":
            if refs:
                for *_, t in b.items:
                    self.reference(t)
            else:
                self.list_block(b.items)
        elif b.kind == "table":
            self.table(b)
        elif b.kind == "figure":
            self.figure(b)
        elif b.kind == "formula":
            self.formula(b.text)
        elif b.kind == "note":
            self.note(b.text)


# ══════════════════════════════════════════════════════════════════════════
# 6. Word COM：更新欄位、存檔、匯出 PDF
# ══════════════════════════════════════════════════════════════════════════
PS1 = r"""
param([Parameter(Mandatory=$true)][string]$Docx, [Parameter(Mandatory=$true)][string]$Pdf)
$ErrorActionPreference = 'Stop'
$word = $null; $doc = $null; $rc = 0
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.ScreenUpdating = $false
    $doc = $word.Documents.Open($Docx, $false, $false, $false)
    [void]$doc.Fields.Update()
    $n = $doc.TablesOfContents.Count
    for ($i = 1; $i -le $n; $i++) { $doc.TablesOfContents.Item($i).Update() }
    $doc.Repaginate()
    for ($i = 1; $i -le $n; $i++) { $doc.TablesOfContents.Item($i).UpdatePageNumbers() }
    foreach ($sec in $doc.Sections) {
        foreach ($hf in $sec.Footers) { [void]$hf.Range.Fields.Update() }
    }
    $doc.Save()
    $doc.ExportAsFixedFormat($Pdf, 17, $false, 0, 0, 1, 1, 0, $true, $true, 1, $true, $true, $false)
    Write-Output ("WORD_PAGES=" + $doc.ComputeStatistics(2))
    Write-Output ("WORD_TOCS=" + $n)
} catch {
    Write-Output ("WORD_ERROR=" + $_.Exception.Message)
    $rc = 1
} finally {
    if ($doc -ne $null) {
        try { $doc.Close([ref]0) } catch { try { $doc.Close() } catch {} }
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc)
    }
    if ($word -ne $null) {
        try { $word.NormalTemplate.Saved = $true } catch {}
        try { $word.Quit([ref]0) } catch { try { $word.Quit() } catch {} }
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    }
    [GC]::Collect(); [GC]::WaitForPendingFinalizers()
}
exit $rc
"""


def word_export(docx: Path, pdf: Path) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="word_export_"))
    ps1 = tmp / "export.ps1"
    ps1.write_text(PS1, encoding="ascii")
    if pdf.exists():
        pdf.unlink()
    cmd = ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(ps1),
           "-Docx", str(docx.resolve()), "-Pdf", str(pdf.resolve())]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    except subprocess.TimeoutExpired:
        return {"error": "Word 匯出逾時（900 秒）；請檢查是否有 Word 對話框卡住（工作管理員結束 WINWORD.EXE）"}
    finally:
        try:
            ps1.unlink()
            tmp.rmdir()
        except OSError:
            pass
    info = {"stdout": r.stdout, "stderr": r.stderr, "rc": r.returncode}
    for line in r.stdout.splitlines():
        if "=" in line and line.split("=", 1)[0] in ("WORD_PAGES", "WORD_TOCS", "WORD_ERROR"):
            k, v = line.split("=", 1)
            info[k] = v.strip()
    if r.returncode != 0 or "WORD_ERROR" in info or not pdf.is_file():
        info["error"] = info.get("WORD_ERROR") or (r.stderr.strip()[-1500:] or f"PowerShell exit {r.returncode}")
    return info


def fix_core_props(docx: Path) -> None:
    """Word 存檔時會把本機 Office 使用者（可能是電子郵件）寫進「最後修改者」；改回作者名稱（只動 docProps/core.xml）。"""
    tmp = docx.with_suffix(".tmp.docx")
    with zipfile.ZipFile(docx) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "docProps/core.xml":
                xml = data.decode("utf-8")
                xml = re.sub(r"<cp:lastModifiedBy>.*?</cp:lastModifiedBy>",
                             f"<cp:lastModifiedBy>{META['authors']}</cp:lastModifiedBy>", xml, flags=re.S)
                data = xml.encode("utf-8")
            zout.writestr(item, data)
    tmp.replace(docx)


def render_pages(pdf: Path, out_dir: Path, dpi: int) -> int:
    import pypdfium2 as pdfium
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("page_*.png"):
        old.unlink()
    doc = pdfium.PdfDocument(str(pdf))
    try:
        for i in range(len(doc)):
            page = doc[i]
            img = page.render(scale=dpi / 72).to_pil()
            img.save(out_dir / f"page_{i + 1:03d}.png")
            page.close()
        return len(doc)
    finally:
        doc.close()


# ══════════════════════════════════════════════════════════════════════════
# 7. 格式自我檢查
# ══════════════════════════════════════════════════════════════════════════
# Word（繁中版）存檔時會把 styleId 改成在地化代碼（a2、afff…），故自我檢查一律以樣式名稱分類
BODY_STYLES = {"body text", "報告清單", "參考文獻條目", "公式"}
CAPTION_STYLES = {"圖說", "表題"}
NOTE_STYLES = {"表註"}
HEADING_STYLES = {"heading 1", "heading 2", "heading 3", "heading 4", "前置標題", "報告目錄標題"}


def _style_chain_size(style, default_pt):
    st = style
    while st is not None:
        if st.font.size is not None:
            return st.font.size.pt
        st = st.base_style
    return default_pt


def self_check(docx: Path, pdf: Path | None, word_info: dict | None) -> tuple[list[str], list[str]]:
    lines, fails = [], []
    d = Document(str(docx))
    dd = d.styles.element.find(qn("w:docDefaults"))
    sz = dd.find(qn("w:rPrDefault")).find(qn("w:rPr")).find(qn("w:sz"))
    default_pt = int(sz.get(qn("w:val"))) / 2 if sz is not None else 10

    # 頁面與邊界
    for k, s in enumerate(d.sections):
        w, h = s.page_width.cm, s.page_height.cm
        m = [s.top_margin.cm, s.bottom_margin.cm, s.left_margin.cm, s.right_margin.cm]
        if abs(w - PAGE_W_CM) > 0.05 or abs(h - PAGE_H_CM) > 0.05 or any(abs(x - MARGIN_CM) > 0.02 for x in m):
            fails.append(f"第 {k + 1} 節頁面 {w:.2f}×{h:.2f} cm、邊界 {m} 不符 A4／2.54 cm")
    s0 = d.sections[-1]
    lines.append(f"DOCX 頁面：{s0.page_width.cm:.2f} × {s0.page_height.cm:.2f} cm（A4）；邊界上下左右 "
                 f"{s0.top_margin.cm:.2f}／{s0.bottom_margin.cm:.2f}／{s0.left_margin.cm:.2f}／{s0.right_margin.cm:.2f} cm；"
                 f"共 {len(d.sections)} 節")

    # 行距：樣式與段落直接格式
    def rule_of(pf):
        r = pf.line_spacing_rule
        return "單行" if r in (None, WD_LINE_SPACING.SINGLE) else str(r)
    nb = d.styles["Normal"].paragraph_format
    bt = d.styles["Body Text"].paragraph_format
    lines.append(f"行距規則：Normal＝{rule_of(nb)}、本文（Body Text）＝{rule_of(bt)}；"
                 f"本文首行縮排 {bt.first_line_indent.pt if bt.first_line_indent else 0:.0f} pt（2 字元）、段後 "
                 f"{bt.space_after.pt if bt.space_after else 0:.0f} pt")
    if rule_of(nb) != "單行" or rule_of(bt) != "單行":
        fails.append("Normal 或 Body Text 不是單行間距")
    # 每個實際使用的段落樣式（含其 basedOn 鏈）與每個段落的直接格式都必須是單行（line=240 auto）
    def line_of(el):
        ppr = el.find(qn("w:pPr"))
        sp = ppr.find(qn("w:spacing")) if ppr is not None else None
        return None if sp is None else (sp.get(qn("w:line")), sp.get(qn("w:lineRule")))

    odd = []
    used_styles = {}
    all_pars = list(d.element.body.iter(qn("w:p")))
    for sec in d.sections:
        for part in (sec.header, sec.footer):
            if not part.is_linked_to_previous:
                all_pars += list(part._element.iter(qn("w:p")))
    for p in all_pars:
        v = line_of(p)
        if v and v[0] is not None and v != ("240", "auto") and v != ("240", None):
            odd.append(f"段落直接格式 line={v}")
        ps = p.find(qn("w:pPr") + "/" + qn("w:pStyle"))
        used_styles[ps.get(qn("w:val")) if ps is not None else "Normal"] = True
    sid_map = {s.style_id: s for s in d.styles if s.type == WD_STYLE_TYPE.PARAGRAPH}
    for sid in used_styles:
        st = sid_map.get(sid)
        while st is not None:
            v = line_of(st.element)
            if v and v[0] is not None:
                if v not in (("240", "auto"), ("240", None)):
                    odd.append(f"樣式 {st.name} line={v}")
                break
            st = st.base_style
    dflt = line_of(dd.find(qn("w:pPrDefault")))
    lines.append(f"非單行間距：{len(odd)} 處（檢查 {len(all_pars)} 段、{len(used_styles)} 個使用中的樣式；"
                 f"文件預設 line={dflt}）")
    if odd or dflt not in (None, ("240", "auto"), ("240", None)):     # None＝Word 預設（單行）
        fails.append(f"非單行間距：{odd[:5]}；文件預設 {dflt}")

    # 字級：依段落樣式分類
    cats = defaultdict(Counter)
    violations = []
    code_runs = 0

    def scan_par(p, in_table):
        nonlocal code_runs
        name = (p.style.name if p.style is not None else "Normal").lower()
        if in_table:
            cat = "表格儲存格"
        elif name in BODY_STYLES:
            cat = "內文（本文、清單、參考文獻、公式）"
        elif name in CAPTION_STYLES:
            cat = "圖說／表題"
        elif name in NOTE_STYLES:
            cat = "表註"
        elif name in HEADING_STYLES:
            cat = "標題"
        elif name.startswith("toc"):
            cat = "目錄條目"
        elif name == "封面":
            cat = "封面"
        else:
            cat = f"其他（{name}）"
        base = _style_chain_size(p.style, default_pt) if p.style is not None else default_pt
        for r in p.runs:
            if not r.text.strip():
                continue
            size = r.font.size.pt if r.font.size is not None else base
            rf = r._r.rPr.rFonts if r._r.rPr is not None else None
            if rf is not None and rf.get(qn("w:ascii")) == FONT_CODE:
                code_runs += 1
                cats["行內程式碼（Consolas）"][size] += len(r.text)
                if cat.startswith("內文") and size < BODY_PT:
                    violations.append(f"{cat}（行內程式碼）{size} pt：{r.text[:30]}")
                continue
            cats[cat][size] += len(r.text)
            if cat.startswith("內文") and size < BODY_PT:
                violations.append(f"{cat} {size} pt：{r.text[:30]}")

    star_bad = []

    def star_check(p):
        """顯著性星號被誤當成粗體／斜體標記時，會出現含「p <」的粗體 run，或殘留反斜線。"""
        for r in p.runs:
            if (r.bold and "p <" in r.text) or "\\*" in r.text:
                star_bad.append(r.text[:40])

    for p in d.paragraphs:
        scan_par(p, False)
        star_check(p)
    for t in d.tables:
        for row in t.rows:
            for c in row.cells:
                for p in c.paragraphs:
                    scan_par(p, True)
                    star_check(p)
    if star_bad:
        fails.append(f"星號或粗體標記異常 {len(star_bad)} 處，例如 {star_bad[:3]}")
    lines.append("各類段落字級（pt：字數）：")
    for cat in sorted(cats):
        dist = "、".join(f"{k:g} pt：{v:,}" for k, v in sorted(cats[cat].items()))
        lines.append(f"  {cat}：{dist}")
    if violations:
        fails.append(f"{len(violations)} 個內文 run 小於 12 pt，例如 {violations[:3]}")
    else:
        lines.append(f"  內文無小於 12 pt 的文字（含行內程式碼 Consolas {code_runs} 處，沿用所在段落字級）")

    # 目錄
    instr = [it.text for it in d.element.body.iter(qn("w:instrText")) if it.text]
    tocs = [t for t in instr if t.strip().startswith("TOC")]
    toc_entries = Counter()
    for p in d.paragraphs:
        if p.style is not None and p.style.name.lower().startswith("toc"):
            if p.text.strip():
                toc_entries[p.style.name.lower()] += 1
    lines.append(f"目錄欄位：{len(tocs)} 個（{'；'.join(t.strip() for t in tocs)}）；"
                 f"更新後條目數：{dict(toc_entries) or '尚未更新'}")
    if not tocs:
        fails.append("找不到 TOC 欄位")
    elif word_info is not None and not toc_entries:
        fails.append("目錄沒有條目（Word 未更新目錄？）")

    # PDF：頁數、尺寸、實際字型與字級
    if pdf is not None and pdf.is_file():
        import pdfplumber
        import pypdfium2 as pdfium
        pdoc = pdfium.PdfDocument(str(pdf))
        sizes = Counter((round(pdoc[i].get_size()[0] / 72 * 2.54, 1), round(pdoc[i].get_size()[1] / 72 * 2.54, 1))
                        for i in range(len(pdoc)))
        npages = len(pdoc)
        try:
            nbook = sum(1 for _ in pdoc.get_toc())
        except Exception:                                   # noqa: BLE001（書籤只是資訊）
            nbook = -1
        pdoc.close()
        lines.append(f"PDF：{npages} 頁；頁面尺寸 {dict(sizes)} cm；標題書籤 {nbook} 個")
        if set(sizes) != {(21.0, 29.7)}:
            fails.append(f"PDF 頁面不是 A4：{dict(sizes)}")
        fonts, fsize = Counter(), Counter()
        toc_page = None
        with pdfplumber.open(str(pdf)) as pp:
            for pi, page in enumerate(pp.pages):
                for chx in page.chars:
                    if not chx["text"].strip():
                        continue
                    fonts[chx["fontname"].split("+")[-1]] += 1
                    fsize[round(chx["size"] * 2) / 2] += 1
                if toc_page is None:
                    txt = page.extract_text() or ""
                    if f"目{IDEO_SPACE}錄" in txt or "目 錄" in txt:
                        toc_page = pi + 1
        total = sum(fsize.values())
        lines.append("PDF 字型（字元數）：" + "、".join(f"{k} {v:,}" for k, v in fonts.most_common()))
        lines.append("PDF 字級分布（pt：占比）：" + "、".join(
            f"{k:g}：{v / total:.1%}" for k, v in sorted(fsize.items(), key=lambda kv: -kv[1]) if v / total >= 0.002))
        need = {"PMingLiU": False, "TimesNewRoman": False, "JhengHei": False}
        for f in fonts:
            for key in need:
                if key.lower() in f.replace(" ", "").replace("-", "").lower() or \
                        (key == "PMingLiU" and "新細明體" in f):
                    need[key] = True
        missing = [k for k, v in need.items() if not v]
        if missing:
            fails.append(f"PDF 未嵌入預期字型：{missing}（可能被替代）")
        lines.append(f"目錄頁：PDF 第 {toc_page} 頁" if toc_page else "目錄頁：PDF 中找不到「目　錄」")
        if not toc_page:
            fails.append("PDF 中找不到目錄頁")
    return lines, fails


# ══════════════════════════════════════════════════════════════════════════
# 8. 主程式
# ══════════════════════════════════════════════════════════════════════════
def resolve_out(name: str) -> Path:
    p = Path(name)
    if not p.is_absolute():
        p = RL / p
    return p.with_suffix("") if p.suffix.lower() in (".docx", ".pdf") else p


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--pages", default=None)
    ap.add_argument("--dpi", type=int, default=90)
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--no-lists", action="store_true")
    ap.add_argument("--toc-first", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_absolute():
        src = (ROOT / src) if (ROOT / src).is_dir() else src.resolve()
    files = sorted(p for p in src.glob("*.md") if not p.name.startswith("_"))
    if not files:
        sys.exit(f"{src} 沒有 .md 章節檔")
    base = resolve_out(args.out)
    base.parent.mkdir(parents=True, exist_ok=True)
    docx_path, pdf_path = base.parent / (base.name + ".docx"), base.parent / (base.name + ".pdf")
    pages_dir = Path(args.pages) if args.pages else base.parent / "pages"

    prob = Problems()
    chapters = [Chapter(f, parse_markdown(f, prob)) for f in files]
    print(f"讀入 {len(files)} 個章節檔：{'、'.join(f.name for f in files)}")
    check_handnumbers(chapters, prob)
    numbers = number_floats(chapters, prob)
    resolve_xrefs(chapters, numbers, prob)

    builder = Builder(prob, lists=not args.no_lists, toc_first=args.toc_first)
    doc = builder.build(chapters)
    if prob.errors:
        print("\n來源錯誤（未輸出）：")
        for e in prob.errors:
            print("  ✗", e)
        for w in prob.warnings:
            print("  ！", w)
        sys.exit(1)
    try:
        doc.save(str(docx_path))
    except PermissionError:
        sys.exit(f"無法寫入 {docx_path}（檔案可能在 Word 中開啟，請先關閉）")
    c = builder.counts
    print(f"寫出 {docx_path}（章 {c['h1']}、節 {c['h2']}、小節 {c['h3']}、段落 {c['para']}、清單 {c['list']}、"
          f"圖 {c['figure']}、表 {c['table']}、參考文獻 {c['ref']}）")

    word_info = None
    if not args.no_pdf:
        print("以 Microsoft Word 更新目錄並匯出 PDF …")
        word_info = word_export(docx_path, pdf_path)
        if "error" in word_info:
            print("Word 匯出失敗：", word_info["error"])
            sys.exit(2)
        fix_core_props(docx_path)
        print(f"寫出 {pdf_path}（Word 統計 {word_info.get('WORD_PAGES')} 頁；更新目錄 {word_info.get('WORD_TOCS')} 個）")
        n = render_pages(pdf_path, pages_dir, args.dpi)
        print(f"寫出 {n} 張頁面 PNG → {pages_dir}（{args.dpi} dpi）")

    print("\n══ 格式自我檢查 ══")
    lines, fails = self_check(docx_path, None if args.no_pdf else pdf_path, word_info)
    for l in lines:
        print(" ", l)
    print(f"  圖 {c['figure']} 張、表 {c['table']} 張（其中 {c['table_10pt']} 張因欄多改用 10 pt）；"
          f"交叉參照 id {len(numbers)} 個")
    if prob.warnings:
        print(f"\n提醒（{len(prob.warnings)}）：")
        for w in prob.warnings:
            print("  ！", w)
    if fails:
        print("\n未通過：")
        for f in fails:
            print("  ✗", f)
        sys.exit(1)
    print("\n全部檢查通過。")


if __name__ == "__main__":
    main()
