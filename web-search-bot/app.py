"""Resume matcher: open http://127.0.0.1:5000 after running `python app.py`."""

import json
import re
import time

import ollama
from flask import Flask, Response, request, send_from_directory

import attachments
import matcher

app = Flask(__name__, static_folder="static")
# Attachments arrive base64-encoded inside the JSON body (about 4/3 of the file size).
app.config["MAX_CONTENT_LENGTH"] = attachments.MAX_FILE_BYTES * 4 // 3 + 1024 * 1024


def event(kind: str, **data) -> str:
    # One JSON object per line (NDJSON) so the browser can read the stream line by line.
    return json.dumps({"type": kind, **data}, ensure_ascii=False) + "\n"


@app.get("/")
def match_page():
    return send_from_directory(app.static_folder, "match.html")


LOOP_MAX_PERIOD = 80   # longest repeated chunk (in characters) the loop check looks for
LOOP_MIN_CHARS = 40    # ...and how much repetition counts as a loop, so short runs like "ๆๆๆ" don't trip it


def loop_start(text: str) -> int | None:
    """If the text ends in the same chunk over and over, return where to cut it (keeping one copy)."""
    text = re.sub(r"\d", "#", text)  # the model also loops by counting: "ปีที่ 3 ปีที่ 4 ปีที่ 5 ..."
    for period in range(4, LOOP_MAX_PERIOD + 1):
        repeats = max(4, -(-LOOP_MIN_CHARS // period))
        if len(text) < period * repeats:
            break
        unit = text[-period:]
        # Only chunks with letters/digits: a long table separator like |------|------| isn't a loop.
        if any(ch.isalnum() for ch in unit) and text.endswith(unit * repeats):
            start = len(text) - period * repeats
            while text[start - period:start] == unit:
                start -= period
            return start + period
    return None


def model_error(model: str, e: Exception) -> str:
    if isinstance(e, ollama.ResponseError):
        if e.status_code == 404:
            return f"ยังไม่มีโมเดล {model} ในเครื่อง รัน ollama pull {model} ก่อน"
        return f"โมเดล {model} ตอบไม่สำเร็จ ({e.error})"
    return f"เรียกโมเดล {model} ไม่สำเร็จ ตรวจว่าเปิด Ollama อยู่ ({e})"  # Ollama not running, etc.


def transcribe_part(data: str):
    """Yields keep-alive pings; returns the image's text and whether the model read it to the end."""
    text, finished, last_sent = "", False, time.monotonic()
    messages = [{"role": "user", "content": attachments.prompt_for(attachments.VISION_MODEL), "images": [data]}]
    for chunk in ollama.chat(model=attachments.VISION_MODEL, messages=messages, stream=True, think=False,
                             options=attachments.VISION_OPTIONS):
        text += chunk["message"]["content"] or ""
        cut = loop_start(text)
        if cut is not None:
            return attachments.clean_transcript(text[:cut]), False
        finished = chunk.get("done_reason") == "stop"  # "length" means it ran out of num_predict
        if time.monotonic() - last_sent > 1:
            yield event("ping")  # Flask only notices the browser left (closed tab) when a write fails
            last_sent = time.monotonic()
    return attachments.clean_transcript(text), finished


def transcribe(image: dict):
    """Have the vision model copy out an image's text. Yields pings; returns the text and if all of it was read.

    The whole page first: Typhoon is trained on whole pages and read the generated test pages almost perfectly
    that way. Only if that doesn't finish (a loop, or past num_predict) try again in three strips.
    """
    page = attachments.fit(image["data"])
    text, finished = yield from transcribe_part(page)
    if finished:
        return text, True
    parts, complete = [], True
    for strip in attachments.strips(page):
        part, done = yield from transcribe_part(strip)
        parts.append(part)
        complete = complete and done
    return attachments.join_strips(parts), complete


@app.post("/api/extract")
def extract():
    """Turn one uploaded file into plain text for a panel on the match page."""
    body = request.get_json(silent=True) or {}

    def generate():
        try:
            images, files = attachments.parse(body.get("attachments"))
        except attachments.AttachmentError as e:
            yield event("error", message=str(e))
            return
        if files:
            yield event("text", text=files[0]["text"])
            return
        if not images:
            yield event("error", message="เลือกไฟล์ก่อน")
            return

        yield event("reading", model=attachments.VISION_MODEL)
        try:
            text, complete = yield from transcribe(images[0])
        except Exception as e:
            yield event("error", message=model_error(attachments.VISION_MODEL, e))
            return
        if text:
            yield event("text", text=text, complete=complete)
        else:
            yield event("error", message="ไม่พบข้อความในรูปนี้ ลองใช้รูปที่ชัดกว่านี้ หรือวางข้อความเอง")

    return Response(generate(), mimetype="application/x-ndjson")


@app.post("/api/interpret")
def interpret():
    """Read the job description once, so every resume is checked against the same list."""
    body = request.get_json(silent=True) or {}
    jd = str(body.get("jd", ""))

    def generate():
        if not jd.strip():
            yield event("error", message="ใส่ job description ก่อนเทียบ")
            return
        if len(jd) > matcher.JD_CHARS:
            yield event("notice", message=f"Job description ยาวเกินไป โมเดลอ่านแค่ {matcher.JD_CHARS:,} ตัวอักษรแรก")
        try:
            result = matcher.interpret_jd(jd)
        except json.JSONDecodeError:
            yield event("error", message="ตีความไม่สำเร็จ ลองตัด job description ให้เหลือเฉพาะส่วนคุณสมบัติ")
            return
        except Exception as e:
            yield event("error", message=model_error(matcher.MODEL, e))
            return
        if result["items"]:
            yield event("interpretation", **result)
        else:
            yield event("error", message="ไม่พบคุณสมบัติผู้สมัครใน job description ลองใส่ส่วนที่ระบุคุณสมบัติมาด้วย")

    return Response(generate(), mimetype="application/x-ndjson")


@app.post("/api/match")
def match():
    body = request.get_json(silent=True) or {}
    # Not stripped: the page highlights quotes by their position in exactly the text it sent.
    resume = str(body.get("resume", ""))
    raw = body.get("requirements")
    items = [str(item) for item in raw][:matcher.MAX_REQUIREMENTS] if isinstance(raw, list) else []
    jd = str(body.get("jd", ""))
    confirmed = body.get("confirmed") is True  # a person said this really is a resume

    def generate():
        if not resume.strip() or not items:
            yield event("error", message="ใส่ทั้งเรซูเม่และ job description ก่อนเทียบ")
            return
        if not confirmed:
            try:
                problem = matcher.document_problem(resume, jd)
            except Exception as e:
                yield event("error", message=model_error(matcher.MODEL, e))
                return
            if problem:
                yield event("not_resume", message=problem)
                return
        if len(resume) > matcher.RESUME_CHARS:
            yield event("notice", message=f"เรซูเม่ยาวเกินไป โมเดลอ่านแค่ {matcher.RESUME_CHARS:,} ตัวอักษรแรก")
        try:
            for index, requirement in enumerate(items):
                yield event("check", index=index, **matcher.check_requirement(requirement, resume))
        except Exception as e:
            yield event("error", message=model_error(matcher.MODEL, e))
            return
        yield event("done")

    return Response(generate(), mimetype="application/x-ndjson")


@app.errorhandler(413)
def too_large(_):
    return {"error": "ไฟล์ใหญ่เกินไป"}, 413


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
