"""Files and images uploaded on the match page, read into plain text."""

import base64
import binascii
import html
import io
import os
import re

from PIL import Image
from pypdf import PdfReader

# paibot is text-only, so images go to Typhoon OCR 1.5 2B, Qwen3-VL-2B trained on Thai documents by SCB 10X:
#   ollama pull hf.co/mradermacher/typhoon-ocr1.5-2b-GGUF:Q4_K_M && ollama cp <that name> typhoon
# On the 50 + 20 generated test pages (thai-doc-ocr) it read 99.7% of fields exactly, against 82% for
# qwen3-vl:2b-instruct, and in about two thirds of the time.
VISION_MODEL = "typhoon"
# Greedy, as it was measured. A Thai page is ~1,500 tokens of Markdown; 4,096 leaves room for a long resume.
# No repeat_penalty: Thai repeats the same tokens a lot and the penalty made models misread text.
VISION_OPTIONS = {"temperature": 0, "num_ctx": 8192, "num_predict": 4096}
VISION_MAX_SIDE = 1800  # Typhoon is trained on pages with an 1800 px longest side
MAX_FILES = 1
MAX_FILE_BYTES = 10 * 1024 * 1024

TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml", ".toml", ".ini", ".log",
    ".html", ".css", ".js", ".ts", ".py", ".java", ".c", ".cpp", ".h", ".cs", ".go", ".rs",
    ".php", ".rb", ".sh", ".sql",
}

# The vision model only copies text out. Asked to answer in Thai straight from a resume picture, the 2B
# model translated names into Thai, guessed the wrong university and looped; asked only to copy the text,
# it got every detail right.
TRANSCRIBE_PROMPT = (
    "Transcribe all the text in the image exactly as written, keeping the original language and line breaks. "
    "Do not translate, summarize or describe. If the image has no readable text, reply with only: NO_TEXT"
)

# Typhoon OCR only works with this exact prompt, from its model card.
TYPHOON_PROMPT = """Extract all text from the image.

Instructions:
- Only return the clean Markdown.
- Do not include any explanation or extra text.
- You must include all information on the page.

Formatting Rules:
- Tables: Render tables using <table>...</table> in clean HTML format.
- Equations: Render equations using LaTeX syntax with inline ($...$) and block ($$...$$).
- Images/Charts/Diagrams: Wrap any clearly defined visual areas (e.g. charts, diagrams, pictures) in:

<figure>
Describe the image's main elements (people, objects, text), note any contextual clues (place, event, culture), mention visible text and its meaning, provide deeper analysis when relevant (especially for financial charts, graphs, or documents), comment on style or architecture if relevant, then give a concise overall summary. Describe in Thai.
</figure>

- Page Numbers: Wrap page numbers in <page_number>...</page_number> (e.g., <page_number>14</page_number>).
- Checkboxes: Use ☐ for unchecked and ☑ for checked boxes."""


def prompt_for(model: str) -> str:
    return TYPHOON_PROMPT if "typhoon" in model else TRANSCRIBE_PROMPT


# Fallback when a whole page doesn't finish (a loop, or past num_predict): read it as 3 overlapping strips.
# With qwen3-vl a whole Thai resume never finished and three strips did; four looped again.
STRIPS = 3
STRIP_OVERLAP = 0.08  # share of a strip repeated at each edge, so a cut doesn't lose a line of text


class AttachmentError(ValueError):
    """A file the bot can't use; the message is shown to the user as is."""


def _pdf_text(name: str, data: bytes) -> str:
    try:
        text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages).strip()
    except Exception:
        raise AttachmentError(f"เปิด {name} ไม่ได้ ไฟล์ PDF อาจเสียหรือติดรหัสผ่าน")
    if not text:
        raise AttachmentError(f"{name} ไม่มีข้อความให้อ่าน น่าจะเป็น PDF ที่สแกนมา ลองแคปหน้าจอแล้วแนบเป็นรูปแทน")
    # Thai PDFs often store สระอำ as nikhahit + sara aa ("ํา"); join it back into "ำ".
    return text.replace("ํา", "ำ")


def clean_transcript(raw: str) -> str:
    """The vision model's transcript as plain text, or "" when it found no text."""
    # Typhoon describes pictures (a resume's photo, a logo) inside <figure>: its own words, not the document's,
    # so a matcher could quote them as evidence. Drop them, and turn its HTML tables and tags into plain lines.
    raw = re.sub(r"<figure>.*?(</figure>|$)", "", raw, flags=re.S)
    raw = re.sub(r"<page_number>.*?</page_number>", "", raw, flags=re.S)
    raw = re.sub(r"</t[dh]>\s*<t[dh][^>]*>", " | ", raw)
    raw = re.sub(r"</?(table|thead|tbody|tr|td|th|br)[^>]*>", "\n", raw)
    # Plain text, as the matcher quotes it: the checker copied "Python for AI & Data Analysis" without the
    # **bold** around "Python" or with &amp; written out, and the quote no longer matched the resume.
    raw = html.unescape(raw).replace("**", "")
    raw = re.sub(r"^[ \t]*#+[ \t]+", "", raw, flags=re.M)  # Markdown headings
    raw = re.sub(r"^([ \t]*)[*+-][ \t]+", r"\1• ", raw, flags=re.M)  # Markdown bullets
    # qwen3-vl tacks NO_TEXT onto the end of real text, mostly on the last strip of a page.
    lines = [line.rstrip() for line in raw.splitlines() if line.strip(" .\"'`").upper() != "NO_TEXT"]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def fit(data: str, max_side: int = VISION_MAX_SIDE) -> str:
    """The image in base64, shrunk so its longest side is at most max_side."""
    image = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    if max(image.size) <= max_side:
        return data
    scale = max_side / max(image.size)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode()


def strips(data: str) -> list[str]:
    """A portrait page as overlapping strips, top to bottom, in base64; anything wider stays whole."""
    image = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    width, height = image.size
    if height <= width:
        return [data]
    step = height / STRIPS
    margin = step * STRIP_OVERLAP
    parts = []
    for i in range(STRIPS):
        buffer = io.BytesIO()
        image.crop((0, max(0, int(i * step - margin)), width, min(height, int((i + 1) * step + margin)))).save(
            buffer, "JPEG", quality=90)
        parts.append(base64.b64encode(buffer.getvalue()).decode())
    return parts


def join_strips(parts: list[str]) -> str:
    """Put the strips back together, dropping lines an earlier strip already read in the overlap."""
    seen, lines = set(), []
    for part in parts:
        fresh = [line for line in part.splitlines() if not line.strip() or line.strip() not in seen]
        seen.update(line.strip() for line in fresh)
        lines += fresh
    return "\n".join(lines).strip()


def parse(raw) -> tuple[list[dict], list[dict]]:
    """Split the request's attachments into images {name, data (base64)} and text files {name, text}."""
    items = raw if isinstance(raw, list) else []
    if len(items) > MAX_FILES:
        raise AttachmentError(f"แนบได้ครั้งละไม่เกิน {MAX_FILES} ไฟล์")

    images, files = [], []
    for item in items:
        name = str(item.get("name") or "ไฟล์")
        kind = str(item.get("type") or "")
        try:
            data = base64.b64decode(str(item.get("data") or ""), validate=True)
        except (binascii.Error, ValueError):
            raise AttachmentError(f"อ่าน {name} ไม่ได้ ลองแนบใหม่อีกครั้ง")
        if len(data) > MAX_FILE_BYTES:
            raise AttachmentError(f"{name} ใหญ่เกิน {MAX_FILE_BYTES // (1024 * 1024)} MB")

        ext = os.path.splitext(name)[1].lower()
        if kind.startswith("image/"):
            images.append({"name": name, "data": base64.b64encode(data).decode()})
        elif ext == ".pdf" or kind == "application/pdf":
            files.append({"name": name, "text": _pdf_text(name, data)})
        elif ext in TEXT_EXTENSIONS or kind.startswith("text/"):
            files.append({"name": name, "text": data.decode("utf-8-sig", errors="replace")})
        else:
            raise AttachmentError(f"ยังอ่าน {name} ไม่ได้ ตอนนี้รองรับรูป, PDF และไฟล์ข้อความ/โค้ด")
    return images, files
