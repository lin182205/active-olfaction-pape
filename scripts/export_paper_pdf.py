"""将论文 Markdown 导出为带中文字体和目录的 A4 PDF。"""

from __future__ import annotations

from html import escape
from pathlib import Path
import re
import unicodedata

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    ListFlowable,
    ListItem,
    LongTable,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper_full.md"
OUTPUT_DIR = ROOT / "output" / "pdf"
OUTPUT = OUTPUT_DIR / "active_olfaction_paper_revised.pdf"

FONT_REGULAR = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")
FONT_NAME = "MicrosoftYaHei"
FONT_BOLD_NAME = "MicrosoftYaHei-Bold"

PAPER_TITLE = "面向间歇羽流的双触须主动嗅觉采样与强化学习搜索方法"


def register_fonts() -> None:
    for path in (FONT_REGULAR, FONT_BOLD):
        if not path.exists():
            raise FileNotFoundError(f"缺少中文字体: {path}")
    pdfmetrics.registerFont(TTFont(FONT_NAME, str(FONT_REGULAR), subfontIndex=0))
    pdfmetrics.registerFont(TTFont(FONT_BOLD_NAME, str(FONT_BOLD), subfontIndex=0))
    pdfmetrics.registerFontFamily(
        FONT_NAME,
        normal=FONT_NAME,
        bold=FONT_BOLD_NAME,
        italic=FONT_NAME,
        boldItalic=FONT_BOLD_NAME,
    )


def draw_first_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setTitle(PAPER_TITLE)
    canvas.setAuthor("论文作者待补")
    canvas.setSubject("双触须主动嗅觉采样与强化学习搜索")
    canvas.restoreState()


def draw_later_pages(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#AAB4C0"))
    canvas.setLineWidth(0.35)
    canvas.line(doc.leftMargin, height - 1.62 * cm, width - doc.rightMargin, height - 1.62 * cm)
    canvas.setFillColor(colors.HexColor("#66717E"))
    canvas.setFont(FONT_NAME, 7.8)
    canvas.drawString(doc.leftMargin, height - 1.40 * cm, PAPER_TITLE)
    canvas.setFont(FONT_NAME, 8.5)
    canvas.drawCentredString(width / 2, 1.25 * cm, str(canvas.getPageNumber()))
    canvas.restoreState()


class PaperDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs) -> None:
        super().__init__(filename, **kwargs)
        first_frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="first_frame",
        )
        body_frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="body_frame",
        )
        self.addPageTemplates(
            [
                PageTemplate(id="First", frames=[first_frame], onPage=draw_first_page),
                PageTemplate(id="Body", frames=[body_frame], onPage=draw_later_pages),
            ]
        )
        self._heading_index = 0

    def beforeDocument(self) -> None:
        """每次目录排版迭代都使用相同的书签编号。"""
        self._heading_index = 0
        super().beforeDocument()

    def afterFlowable(self, flowable) -> None:
        if not isinstance(flowable, Paragraph):
            return
        level_map = {"Heading1CN": 0, "Heading2CN": 1, "Heading3CN": 2}
        level = level_map.get(flowable.style.name)
        if level is None:
            return
        text = flowable.getPlainText()
        key = f"heading-{self._heading_index}"
        self._heading_index += 1
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=False)
        self.notify("TOCEntry", (level, text, self.page, key))


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "BodyCN",
        parent=base["BodyText"],
        fontName=FONT_NAME,
        fontSize=10.2,
        leading=17.2,
        alignment=TA_JUSTIFY,
        firstLineIndent=2 * 10.2,
        spaceAfter=6,
        textColor=colors.HexColor("#20262D"),
        wordWrap="CJK",
        allowWidows=0,
        allowOrphans=0,
    )
    return {
        "body": body,
        "title": ParagraphStyle(
            "TitleCN",
            fontName=FONT_BOLD_NAME,
            fontSize=22,
            leading=33,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#172A3A"),
            spaceAfter=24,
            wordWrap="CJK",
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCN",
            fontName=FONT_NAME,
            fontSize=12,
            leading=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#566573"),
            spaceAfter=12,
            wordWrap="CJK",
        ),
        "meta": ParagraphStyle(
            "MetaCN",
            fontName=FONT_NAME,
            fontSize=11,
            leading=24,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#34495E"),
            spaceAfter=4,
            wordWrap="CJK",
        ),
        "note": ParagraphStyle(
            "NoteCN",
            fontName=FONT_NAME,
            fontSize=9,
            leading=16,
            alignment=TA_LEFT,
            leftIndent=0.5 * cm,
            rightIndent=0.5 * cm,
            borderWidth=0.6,
            borderColor=colors.HexColor("#B9C7D5"),
            borderPadding=9,
            backColor=colors.HexColor("#F5F8FB"),
            textColor=colors.HexColor("#536270"),
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "Heading1CN",
            fontName=FONT_BOLD_NAME,
            fontSize=16,
            leading=24,
            spaceBefore=14,
            spaceAfter=11,
            textColor=colors.HexColor("#16324F"),
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "Heading2CN",
            fontName=FONT_BOLD_NAME,
            fontSize=13,
            leading=20,
            spaceBefore=12,
            spaceAfter=7,
            textColor=colors.HexColor("#23527C"),
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "h3": ParagraphStyle(
            "Heading3CN",
            fontName=FONT_BOLD_NAME,
            fontSize=11.2,
            leading=18,
            spaceBefore=9,
            spaceAfter=5,
            textColor=colors.HexColor("#315D7C"),
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "bullet": ParagraphStyle(
            "BulletCN",
            parent=body,
            firstLineIndent=0,
            leftIndent=0,
            spaceAfter=3,
        ),
        "caption": ParagraphStyle(
            "CaptionCN",
            parent=body,
            firstLineIndent=0,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "equation": ParagraphStyle(
            "EquationCN",
            parent=body,
            firstLineIndent=0,
            alignment=TA_CENTER,
            fontSize=9.4,
            leading=15,
            leftIndent=0.35 * cm,
            rightIndent=0.35 * cm,
            spaceBefore=4,
            spaceAfter=7,
        ),
        "quote": ParagraphStyle(
            "QuoteCN",
            parent=body,
            firstLineIndent=0,
            leftIndent=0.4 * cm,
            rightIndent=0.2 * cm,
            borderColor=colors.HexColor("#8FA7BA"),
            borderWidth=0,
            borderLeft=2,
            borderPadding=6,
            backColor=colors.HexColor("#F7F9FB"),
        ),
        "code": ParagraphStyle(
            "CodeCN",
            fontName=FONT_NAME,
            fontSize=8.2,
            leading=13,
            leftIndent=0.25 * cm,
            rightIndent=0.25 * cm,
            borderPadding=7,
            borderWidth=0.4,
            borderColor=colors.HexColor("#D4DAE0"),
            backColor=colors.HexColor("#F5F6F7"),
            textColor=colors.HexColor("#263238"),
        ),
        "table": ParagraphStyle(
            "TableCellCN",
            fontName=FONT_NAME,
            fontSize=7.2,
            leading=10.5,
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
        "table_header": ParagraphStyle(
            "TableHeaderCN",
            fontName=FONT_BOLD_NAME,
            fontSize=7.2,
            leading=10.5,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17324D"),
            wordWrap="CJK",
        ),
        "toc_title": ParagraphStyle(
            "TOCTitleCN",
            fontName=FONT_BOLD_NAME,
            fontSize=18,
            leading=26,
            alignment=TA_CENTER,
            spaceAfter=18,
            textColor=colors.HexColor("#16324F"),
        ),
    }


def strip_markdown_emphasis(text: str) -> str:
    text = text.strip()
    if text.startswith("**") and text.endswith("**"):
        text = text[2:-2]
    elif text.startswith("*") and text.endswith("*"):
        text = text[1:-1]
    return text


def replace_braced_command(tex: str, command: str, arity: int) -> str:
    """替换带花括号参数的 LaTeX 命令，支持参数内嵌套花括号。"""
    search_from = 0
    while True:
        start = tex.find(command, search_from)
        if start < 0:
            return tex
        cursor = start + len(command)
        arguments: list[str] = []
        valid = True
        for _ in range(arity):
            while cursor < len(tex) and tex[cursor].isspace():
                cursor += 1
            if cursor >= len(tex) or tex[cursor] != "{":
                valid = False
                break
            depth = 0
            end = cursor
            while end < len(tex):
                if tex[end] == "{":
                    depth += 1
                elif tex[end] == "}":
                    depth -= 1
                    if depth == 0:
                        arguments.append(tex[cursor + 1 : end])
                        cursor = end + 1
                        break
                end += 1
            else:
                valid = False
                break
        if not valid or len(arguments) != arity:
            search_from = start + len(command)
            continue
        if arity == 2:
            replacement = f"(({arguments[0]})/({arguments[1]}))"
        else:
            replacement = f"√({arguments[0]})"
        tex = tex[:start] + replacement + tex[cursor:]
        search_from = max(0, start - 1)


def tex_to_markup(tex: str) -> str:
    """把常用 LaTeX 数学语法转成 ReportLab 可显示的近似排版。"""
    tex = tex.strip()
    tex = replace_braced_command(tex, r"\frac", 2)
    tex = replace_braced_command(tex, r"\sqrt", 1)
    replacements = {
        r"\mathbb{R}": "ℝ",
        r"\mathbb E": "E",
        r"\mathsf{T}": "T",
        r"\nabla": "grad ",
        r"\partial": "d",
        r"\int": "int ",
        r"\ell": "ℓ",
        r"\ldots": "…",
        r"\sim": "∼",
        r"\exp": "exp",
        r"\log": "log",
        r"\cos": "cos",
        r"\sin": "sin",
        r"\tanh": "tanh",
        r"\min": "min",
        r"\max": "max",
        r"\parallel": "∥",
        r"\perp": "⊥",
        r"\theta": "θ",
        r"\phi": "φ",
        r"\varphi": "φ",
        r"\psi": "ψ",
        r"\omega": "ω",
        r"\alpha": "α",
        r"\beta": "β",
        r"\gamma": "γ",
        r"\delta": "δ",
        r"\epsilon": "ε",
        r"\varepsilon": "ε",
        r"\lambda": "λ",
        r"\mu": "μ",
        r"\pi": "π",
        r"\rho": "ρ",
        r"\sigma": "σ",
        r"\tau": "τ",
        r"\Delta": "Δ",
        r"\Omega": "Ω",
        r"\sum": "Σ",
        r"\prod": "Π",
        r"\infty": "∞",
        r"\in": "∈",
        r"\leq": "≤",
        r"\geq": "≥",
        r"\neq": "≠",
        r"\approx": "≈",
        r"\propto": "∝",
        r"\rightarrow": "→",
        r"\times": "×",
        r"\cdot": "·",
        r"\odot": "⊙",
        r"\pm": "±",
        r"\mid": "|",
        r"\lVert": "‖",
        r"\rVert": "‖",
        r"\top": "T",
    }
    for source, target in replacements.items():
        tex = tex.replace(source, target)
    for _ in range(4):
        tex = re.sub(
            r"\\(?:operatorname|mathrm|text|boldsymbol|mathcal|mathbb|widehat|widetilde|overline)\{([^{}]*)\}",
            r"\1",
            tex,
        )
    tex = re.sub(r"\\(?:mathcal|mathbb|widehat|widetilde|overline|mathrm)\s*", "", tex)
    tex = tex.replace(r"\left", "").replace(r"\right", "")
    tex = tex.replace(r"\qquad", "   ").replace(r"\quad", "  ")
    tex = tex.replace(r"\,", " ").replace(r"\!", "").replace(r"\;", " ")
    tex = tex.replace(r"\begin{aligned}", "").replace(r"\end{aligned}", "")
    tex = tex.replace(r"\begin{cases}", "{").replace(r"\end{cases}", "")
    tex = tex.replace("&", "").replace(r"\\", " ; ")
    tex = escape(tex, quote=False)
    for _ in range(3):
        tex = re.sub(r"_\{([^{}]*)\}", r"<sub>\1</sub>", tex)
        tex = re.sub(r"\^\{([^{}]*)\}", r"<super>\1</super>", tex)
    tex = re.sub(r"_([A-Za-z0-9α-ωΑ-Ω*]+)", r"<sub>\1</sub>", tex)
    tex = re.sub(r"\^([A-Za-z0-9α-ωΑ-Ω*]+)", r"<super>\1</super>", tex)
    tex = tex.replace("{", "").replace("}", "")
    tex = re.sub(r"\\[A-Za-z]+", "", tex)
    return tex


def inline_markup(text: str) -> str:
    text = text.replace("\\[", "[").replace("\\]", "]")
    text = text.replace("\\_", "_").replace("\\<", "&lt;").replace("\\>", "&gt;")
    math_fragments: list[str] = []

    def stash_math(match: re.Match[str]) -> str:
        math_fragments.append(tex_to_markup(match.group(1)))
        return f"MATHFRAGMENT{len(math_fragments) - 1}TOKEN"

    text = re.sub(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", stash_math, text)
    escaped = escape(text, quote=False)
    escaped = re.sub(r"`([^`]+)`", r'<font color="#2F5D7C">\1</font>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", escaped)
    for index, fragment in enumerate(math_fragments):
        escaped = escaped.replace(f"MATHFRAGMENT{index}TOKEN", fragment)
    return escaped


def display_width(text: str) -> float:
    clean = re.sub(r"[*`\\$]", "", text)
    width = 0.0
    for char in clean:
        width += 1.0 if unicodedata.east_asian_width(char) in {"W", "F", "A"} else 0.58
    return max(width, 1.0)


def table_widths(rows: list[list[str]], total_width: float) -> list[float]:
    cols = max(len(row) for row in rows)
    if cols >= 6:
        return [total_width / cols] * cols
    maxima = []
    for col in range(cols):
        lengths = [display_width(row[col]) if col < len(row) else 1.0 for row in rows]
        maxima.append(min(max(lengths), 30.0))
    weights = [max(5.0, value ** 0.72) for value in maxima]
    weight_sum = sum(weights)
    return [total_width * value / weight_sum for value in weights]


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    index = start
    while index < len(lines) and lines[index].strip().startswith("|"):
        row = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
        if not all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in row):
            rows.append(row)
        index += 1
    return rows, index


def make_table(rows: list[list[str]], styles: dict[str, ParagraphStyle], width: float):
    cols = max(len(row) for row in rows)
    normalized = [row + [""] * (cols - len(row)) for row in rows]
    data = []
    for row_index, row in enumerate(normalized):
        style = styles["table_header"] if row_index == 0 else styles["table"]
        data.append([Paragraph(inline_markup(cell), style) for cell in row])
    table = LongTable(
        data,
        colWidths=table_widths(normalized, width),
        repeatRows=1,
        splitByRow=1,
        hAlign="CENTER",
        spaceBefore=4,
        spaceAfter=9,
    )
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DFEAF3")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17324D")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AEBCC8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for row_index in range(1, len(data)):
        if row_index % 2 == 0:
            commands.append(
                ("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#F7F9FB"))
            )
    table.setStyle(commands)
    return table


def make_toc(styles: dict[str, ParagraphStyle]) -> TableOfContents:
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            "TOCLevel1CN",
            fontName=FONT_BOLD_NAME,
            fontSize=10.5,
            leading=18,
            leftIndent=0,
            firstLineIndent=0,
            spaceBefore=4,
            textColor=colors.HexColor("#263746"),
        ),
        ParagraphStyle(
            "TOCLevel2CN",
            fontName=FONT_NAME,
            fontSize=9.5,
            leading=16,
            leftIndent=0.55 * cm,
            firstLineIndent=0,
            textColor=colors.HexColor("#455A64"),
        ),
        ParagraphStyle(
            "TOCLevel3CN",
            fontName=FONT_NAME,
            fontSize=8.5,
            leading=14,
            leftIndent=1.1 * cm,
            firstLineIndent=0,
            textColor=colors.HexColor("#60717D"),
        ),
    ]
    return toc


def markdown_story(text: str, styles: dict[str, ParagraphStyle], content_width: float):
    lines = text.splitlines()
    story = []
    index = 0

    front: list[str] = []
    while index < len(lines) and not lines[index].startswith("# "):
        if lines[index].strip():
            front.append(lines[index].strip())
        index += 1

    story.append(Spacer(1, 3.1 * cm))
    if front:
        story.append(Paragraph(inline_markup(strip_markdown_emphasis(front[0])), styles["title"]))
    if len(front) > 1:
        story.append(Paragraph(inline_markup(strip_markdown_emphasis(front[1])), styles["subtitle"]))
    story.append(Spacer(1, 1.0 * cm))
    for item in front[2:6]:
        story.append(Paragraph(inline_markup(strip_markdown_emphasis(item)), styles["meta"]))
    if len(front) > 6:
        story.append(Spacer(1, 1.0 * cm))
        for item in front[6:]:
            story.append(Paragraph(inline_markup(strip_markdown_emphasis(item)), styles["note"]))

    story.extend([NextPageTemplate("Body"), PageBreak()])
    story.append(Paragraph("目录", styles["toc_title"]))
    story.append(make_toc(styles))
    story.append(PageBreak())

    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        if not stripped:
            index += 1
            continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading:
            level = len(heading.group(1))
            story.append(Paragraph(inline_markup(heading.group(2)), styles[f"h{level}"]))
            index += 1
            continue

        if stripped.startswith("|"):
            rows, index = parse_table(lines, index)
            if rows:
                story.append(make_table(rows, styles, content_width))
            continue

        if stripped.startswith("```"):
            index += 1
            code_lines = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            index += 1
            story.append(Preformatted("\n".join(code_lines), styles["code"]))
            story.append(Spacer(1, 5))
            continue

        if stripped == "$$":
            index += 1
            equation_lines = []
            while index < len(lines) and lines[index].strip() != "$$":
                equation_lines.append(lines[index].strip())
                index += 1
            if index < len(lines):
                index += 1
            story.append(Paragraph(tex_to_markup(" ".join(equation_lines)), styles["equation"]))
            continue

        if stripped.startswith("- "):
            items = []
            while index < len(lines):
                candidate = lines[index].strip()
                if candidate.startswith("- "):
                    items.append(
                        ListItem(
                            Paragraph(inline_markup(candidate[2:]), styles["bullet"]),
                            leftIndent=14,
                        )
                    )
                    index += 1
                    while index < len(lines) and not lines[index].strip():
                        index += 1
                else:
                    break
            story.append(
                ListFlowable(
                    items,
                    bulletType="bullet",
                    start="circle",
                    leftIndent=18,
                    bulletFontName=FONT_NAME,
                    bulletFontSize=7,
                    bulletColor=colors.HexColor("#315D7C"),
                    spaceAfter=6,
                )
            )
            continue

        if stripped.startswith(">"):
            quote_lines = []
            while index < len(lines) and (lines[index].strip().startswith(">") or not lines[index].strip()):
                candidate = lines[index].strip()
                if candidate.startswith(">"):
                    content = candidate[1:].strip()
                    if content:
                        quote_lines.append(content)
                index += 1
            for quote in quote_lines:
                quote = re.sub(r"^(\d+)\\\.\s*", r"\1. ", quote)
                story.append(Paragraph(inline_markup(quote), styles["quote"]))
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if not candidate:
                break
            if (
                re.match(r"^(#{1,3})\s+", candidate)
                or candidate == "$$"
                or candidate.startswith(("|", "- ", ">", "```"))
            ):
                break
            paragraph_lines.append(candidate)
            index += 1
        paragraph_text = " ".join(paragraph_lines)
        paragraph_style = styles["caption"] if re.match(r"^表\d+[-－]\d+\s", paragraph_text) else styles["body"]
        story.append(Paragraph(inline_markup(paragraph_text), paragraph_style))

    return story


def main() -> None:
    register_fonts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    text = SOURCE.read_text(encoding="utf-8")
    doc = PaperDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=2.45 * cm,
        rightMargin=2.35 * cm,
        topMargin=2.05 * cm,
        bottomMargin=1.90 * cm,
        title=PAPER_TITLE,
        author="论文作者待补",
    )
    styles = build_styles()
    story = markdown_story(text, styles, doc.width)
    doc.multiBuild(story)
    print(OUTPUT)


if __name__ == "__main__":
    main()
