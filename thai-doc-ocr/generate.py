"""Make fictional resumes and job descriptions as images and PDFs, each with its exact text and JSON answer.

    python generate.py --split test --resumes 36 --jobs 14

Everything is rendered in a browser because Thai needs real text shaping: drawn with plain Pillow, vowels and
tone marks land in the wrong place. All people, companies and contact details are made up.
"""

import argparse
import html
import io
import json
import random
from pathlib import Path

from PIL import Image, ImageFilter
from playwright.sync_api import sync_playwright

from pools import (BENEFITS, COMPANIES, FIRST_EN, FIRST_TH, LAST_EN, LAST_TH, LONG_COMPANIES, LONG_DUTIES, LONG_SCHOOLS,
                   PROVINCES, ROLES, SCHOOLS, TRAITS, VARIED_ABILITIES, VARIED_OTHER)

HERE = Path(__file__).parent
FONTS = ["Sarabun", "Prompt", "Kanit", "Noto Sans Thai", "Noto Serif Thai", "Bai Jamjuree", "IBM Plex Sans Thai"]
FONT_LINK = "https://fonts.googleapis.com/css2?" + "&".join(
    f"family={f.replace(' ', '+')}:wght@400;600" for f in FONTS) + "&display=swap"
RESUME_LAYOUTS = ["classic", "sidebar", "banner", "compact"]
JOB_LAYOUTS = ["plain", "poster", "boxed"]
LANGUAGES = ["th", "th", "en", "mixed"]  # mostly Thai, since that's the hard part
SCANNED_SHARE = 1 / 3

LABELS = {
    "th": {"summary": "สรุปประวัติ", "experience": "ประสบการณ์ทำงาน", "education": "การศึกษา", "skills": "ทักษะ",
           "languages": "ภาษา", "contact": "ติดต่อ", "present": "ปัจจุบัน", "duties": "หน้าที่ความรับผิดชอบ",
           "requirements": "คุณสมบัติผู้สมัคร", "benefits": "สวัสดิการ", "apply": "สนใจสมัครส่งประวัติมาที่",
           "salary": "เงินเดือน", "location": "สถานที่ทำงาน", "hiring": "รับสมัครงาน", "type": "รูปแบบงาน"},
    "en": {"summary": "Summary", "experience": "Work Experience", "education": "Education", "skills": "Skills",
           "languages": "Languages", "contact": "Contact", "present": "Present", "duties": "Responsibilities",
           "requirements": "Requirements", "benefits": "Benefits", "apply": "Send your resume to",
           "salary": "Salary", "location": "Location", "hiring": "We are hiring", "type": "Employment type"},
}
LABELS["mixed"] = LABELS["en"]  # English headings over Thai content, as many Thai resumes do

CSS = """
*{box-sizing:border-box} html,body{margin:0;background:#fff;color:#222}
body{width:794px;min-height:1123px;font-family:var(--font),sans-serif;font-size:var(--size);line-height:1.55}
h1{font-size:2em;margin:0;font-weight:600} h2{font-size:1.1em;margin:18px 0 6px;font-weight:600;color:var(--accent);
border-bottom:1px solid #ccc;padding-bottom:2px} h3{font-size:1em;margin:8px 0 0;font-weight:600}
p{margin:2px 0} ul{margin:2px 0;padding-left:1.2em} .muted{color:#555}
.classic{padding:48px 56px} .compact{padding:32px 40px;font-size:calc(var(--size) - 1px)}
.compact h2{margin-top:10px}
.sidebar{display:grid;grid-template-columns:250px 1fr;min-height:1123px}
.sidebar .side{background:var(--tint);padding:40px 22px} .sidebar .main{padding:40px 32px}
.banner header{background:var(--accent);color:#fff;padding:36px 56px} .banner .body{padding:12px 56px 40px}
.banner header .muted{color:#eee}
.plain{padding:56px} .boxed{padding:40px} .boxed section{border:1px solid #bbb;border-radius:8px;padding:6px 16px;margin-top:12px}
.boxed h2{border:0;margin-top:6px}
.poster{background:#16232b;color:#eef;min-height:1123px;padding:56px} .poster h1{font-size:2.6em}
.poster h2{color:#ffd24d;border-color:#456} .poster .muted{color:#b8c6cf}
"""


def hold_out(pool: list, split: str) -> list:
    """Every fourth entry is kept for the test set, so training never sees a test name, school or company."""
    return [item for i, item in enumerate(pool) if (i % 4 == 0) == (split == "test")]


def in_language(value, language: str, rng: random.Random):
    """Pools hold (Thai, English) pairs; mixed documents pick per item, the way real ones do."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return value[1] if language == "en" or (language == "mixed" and rng.random() < 0.5) else value[0]
    return value


def year(ce: int, language: str) -> str:
    return str(ce + 543) if language == "th" else str(ce)  # Thai documents count in the Buddhist Era


def make_resume(rng: random.Random, split: str, language: str, long: bool = False) -> dict:
    role = rng.choice(ROLES)
    use_thai_name = language != "en"
    first = rng.choice(hold_out(FIRST_TH if use_thai_name else FIRST_EN, split))
    last = rng.choice(hold_out(LAST_TH if use_thai_name else LAST_EN, split))
    pick = lambda value: in_language(value, language, rng)

    # Random draws stay in their original order on the short path, so the existing test set regenerates unchanged.
    if long:  # a career of 4-5 jobs, closer to the 2,000+ character resumes people really send
        start = rng.randint(2006, 2011)
        employers = rng.sample(hold_out(COMPANIES, split) + hold_out(LONG_COMPANIES, split), 5)
        count = rng.randint(4, 5)
    else:
        start = rng.randint(2012, 2022)
        employers = rng.sample(hold_out(COMPANIES, split), 3)
        count = rng.randint(1, 3)
    jobs, cursor = [], start
    for i in range(count):
        end = cursor + rng.randint(1, 3)
        current = end >= 2025 or i == (count - 1 if long else 2)
        details = lambda: rng.sample(role["duties"], rng.randint(2, 3)) + (rng.sample(LONG_DUTIES, 2) if long else [])
        jobs.append({
            "title": pick(role["title"]),
            "company": pick(employers[i]),
            "start": year(cursor, language),
            "end": None if current else year(end, language),
            "details": [pick(d) for d in details()],  # drawn here, after title and company, as the original code did
        })
        if current:
            break
        cursor = end
    jobs.reverse()

    def education() -> list[dict]:
        degrees = [{
            "degree": pick(("ปริญญาตรี", "Bachelor's degree")),
            "field": pick(rng.choice(role["fields"])),
            "school": pick(rng.choice(hold_out(SCHOOLS, split))),
            "year": year(start - 1, language),
        }]
        if long:
            degrees.insert(0, {
                "degree": pick(("ปริญญาโท", "Master's degree")),
                "field": pick(rng.choice(role["fields"])),
                "school": pick(rng.choice(hold_out(LONG_SCHOOLS, split))),
                "year": year(start + rng.randint(3, 6), language),
            })
        return degrees

    return {
        "doc_type": "resume",
        "name": f"{first} {last}",
        "position": pick(role["title"]),
        "email": f"{rng.choice(['contact', 'me', 'hello', 'work'])}{rng.randint(10, 99)}@example.com",
        "phone": f"08{rng.randint(1, 9)}-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}",
        "location": pick(rng.choice(PROVINCES)),
        "summary": " ".join(pick(s) for s in role["summaries"]) if long else pick(rng.choice(role["summaries"])),
        "skills": list(role["skills"]) if long else rng.sample(role["skills"], rng.randint(4, min(6, len(role["skills"])))),
        "languages": [pick(("ภาษาไทย", "Thai")), pick(("ภาษาอังกฤษ", "English")) + f" (TOEIC {rng.randint(55, 95) * 10})"],
        "education": education(),  # called here, after the fields above, as the original code drew it
        "experience": jobs,
    }


def make_job(rng: random.Random, split: str, language: str, long: bool = False) -> dict:
    role = rng.choice(ROLES)
    pick = lambda value: in_language(value, language, rng)
    low = rng.randint(15, 45)
    salary = f"{low},000 - {low + rng.randint(5, 20)},000" + (" บาท" if language == "th" else " THB")
    return {
        "doc_type": "job_description",
        "position": pick(role["title"]),
        "company": pick(rng.choice(hold_out(COMPANIES, split))),
        "location": pick(rng.choice(PROVINCES)),
        "employment_type": pick(rng.choice([("งานประจำ", "Full-time"), ("สัญญาจ้าง", "Contract")])),
        "salary": salary,
        "requirements": {
            "skills": [pick(("ใช้ {} ได้ดี", "Good knowledge of {}")).format(s) for s in rng.sample(role["skills"], 3)],
            "abilities": [pick(("มีประสบการณ์อย่างน้อย {} ปี", "At least {} years of experience")).format(rng.randint(1, 5))],
            "personality": [pick(t) for t in rng.sample(TRAITS, 4 if long else 2)],
            "other": [pick(rng.choice(role["fields_required"]))],
        },
        "responsibilities": [pick(d) for d in rng.sample(role["duties"], 4 if long else 3) + (rng.sample(LONG_DUTIES, 4) if long else [])],
        "benefits": [pick(b) for b in rng.sample([("ประกันสุขภาพ", "Health insurance"), ("โบนัสประจำปี", "Annual bonus"),
                                                  ("ทำงานแบบไฮบริด", "Hybrid work"), ("ประกันสังคม", "Social security")], 4 if long else 2)],
        "contact": f"hr{rng.randint(1, 9)}@example.com",
    }


def make_varied_job(rng: random.Random, split: str, language: str, long: bool = False) -> dict:
    """A job description with a different number of items in each requirement group. With fixed counts (3 skills,
    1 ability, 2 traits, 1 other) a model trained on them sorted the shared requirement list by position, not
    meaning. Job_html also shuffles that list when given an rng."""
    role = rng.choice(ROLES)
    pick = lambda value: in_language(value, language, rng)
    low = rng.randint(15, 45)
    salary = f"{low},000 - {low + rng.randint(5, 20)},000" + (" บาท" if language == "th" else " THB")
    experience = pick(("มีประสบการณ์อย่างน้อย {} ปี", "At least {} years of experience")).format(rng.randint(1, 5))
    return {
        "doc_type": "job_description",
        "position": pick(role["title"]),
        "company": pick(rng.choice(hold_out(COMPANIES, split))),
        "location": pick(rng.choice(PROVINCES)),
        "employment_type": pick(rng.choice([("งานประจำ", "Full-time"), ("สัญญาจ้าง", "Contract")])),
        "salary": salary,
        "requirements": {
            "skills": [pick(("ใช้ {} ได้ดี", "Good knowledge of {}")).format(s)
                       for s in rng.sample(role["skills"], rng.randint(1, 4))],
            "abilities": [experience] + [pick(a) for a in rng.sample(VARIED_ABILITIES, rng.randint(0, 2))],
            "personality": [pick(t) for t in rng.sample(TRAITS, rng.randint(0, 3))],
            "other": [pick(rng.choice(role["fields_required"]))] + [pick(o) for o in rng.sample(VARIED_OTHER, rng.randint(0, 2))],
        },
        "responsibilities": [pick(d) for d in rng.sample(role["duties"], rng.randint(2, 4))],
        "benefits": [pick(b) for b in rng.sample(BENEFITS, rng.randint(1, 4))],
        "contact": f"hr{rng.randint(1, 9)}@example.com",
    }


def bullets(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{html.escape(i)}</li>" for i in items) + "</ul>"


def resume_html(doc: dict, layout: str, language: str) -> str:
    t = LABELS[language]
    e = html.escape
    head = f"<h1>{e(doc['name'])}</h1><p class=muted>{e(doc['position'])}</p>"
    contact = f"<p>{e(doc['email'])}</p><p>{e(doc['phone'])}</p><p>{e(doc['location'])}</p>"
    summary = f"<h2>{t['summary']}</h2><p>{e(doc['summary'])}</p>"
    skills = f"<h2>{t['skills']}</h2>{bullets(doc['skills'])}"
    spoken = f"<h2>{t['languages']}</h2>{bullets(doc['languages'])}"
    education = f"<h2>{t['education']}</h2>" + "".join(
        f"<h3>{e(x['degree'])} {e(x['field'])}</h3><p>{e(x['school'])}</p><p class=muted>{e(x['year'])}</p>"
        for x in doc["education"])
    experience = f"<h2>{t['experience']}</h2>" + "".join(
        f"<h3>{e(x['title'])}</h3><p>{e(x['company'])}</p>"
        f"<p class=muted>{e(x['start'])} - {e(x['end'] or t['present'])}</p>{bullets(x['details'])}"
        for x in doc["experience"])
    if layout == "sidebar":
        return (f"<div class=sidebar><div class=side>{head}<h2>{t['contact']}</h2>{contact}{skills}{spoken}</div>"
                f"<div class=main>{summary}{experience}{education}</div></div>")
    if layout == "banner":
        return f"<div class=banner><header>{head}{contact}</header><div class=body>{summary}{experience}{education}{skills}{spoken}</div></div>"
    return f"<div class={layout}>{head}{contact}{summary}{experience}{education}{skills}{spoken}</div>"


def job_html(doc: dict, layout: str, language: str, shuffle: random.Random | None = None) -> str:
    t = LABELS[language]
    e = html.escape
    r = doc["requirements"]
    wanted = r["skills"] + r["abilities"] + r["personality"] + r["other"]
    if shuffle:  # real job ads don't list requirements group by group
        shuffle.shuffle(wanted)
    parts = [
        f"<section><h2>{t['duties']}</h2>{bullets(doc['responsibilities'])}</section>",
        f"<section><h2>{t['requirements']}</h2>{bullets(wanted)}</section>",
        f"<section><h2>{t['benefits']}</h2>{bullets(doc['benefits'])}</section>",
    ]
    return (f"<div class={layout}><p class=muted>{t['hiring']} - {e(doc['company'])}</p><h1>{e(doc['position'])}</h1>"
            f"<p>{t['type']}: {e(doc['employment_type'])}</p><p>{t['location']}: {e(doc['location'])}</p>"
            f"<p>{t['salary']}: {e(doc['salary'])}</p>{''.join(parts)}<p>{t['apply']} {e(doc['contact'])}</p></div>")


def page_html(body: str, font: str, size: int, accent: str, tint: str) -> str:
    return (f'<!doctype html><html lang=th><head><meta charset=utf-8><link rel=stylesheet href="{FONT_LINK}">'
            f"<style>:root{{--font:'{font}';--size:{size}px;--accent:{accent};--tint:{tint}}}{CSS}</style></head>"
            f"<body>{body}</body></html>")


def scan_effect(image: Image.Image, rng: random.Random) -> Image.Image:
    """Make a clean render look scanned: a slight tilt, softness, grey paper and heavy JPEG compression."""
    image = image.convert("L").rotate(rng.uniform(-1.2, 1.2), fillcolor=235, resample=Image.BICUBIC)
    image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 0.8)))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=rng.randint(45, 65))
    return Image.open(buffer).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--resumes", type=int, default=36)
    parser.add_argument("--jobs", type=int, default=14)
    parser.add_argument("--long", action="store_true", help="long documents (4-5 jobs, two degrees) in data/<split>_long")
    parser.add_argument("--varied", action="store_true",
                        help="job descriptions with varied, shuffled requirement groups, in data/<split>_varied")
    args = parser.parse_args()

    # Different seeds, on top of the held-out names, so no document appears in both splits.
    rng = random.Random({"test": 2569, "train": 7}[args.split] + (100 if args.long else 0) + (200 if args.varied else 0))
    folder = HERE / "data" / (args.split + ("_long" if args.long else "") + ("_varied" if args.varied else ""))
    make_job_doc = make_varied_job if args.varied else make_job
    (folder / "images").mkdir(parents=True, exist_ok=True)
    (folder / "pdf").mkdir(exist_ok=True)

    kinds = ["resume"] * args.resumes + ["job"] * args.jobs
    records = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 794, "height": 1123}, device_scale_factor=1.6)
        for number, kind in enumerate(kinds, 1):
            language = rng.choice(LANGUAGES)
            layout = rng.choice(RESUME_LAYOUTS if kind == "resume" else JOB_LAYOUTS)
            font = rng.choice(FONTS)
            size = rng.randint(14, 16)
            accent = rng.choice(["#1f5f8b", "#7a3b69", "#2e6b4f", "#8a4b12", "#333333"])
            tint = rng.choice(["#eef3f7", "#f5eff4", "#eef5f0", "#f6f1ea"])
            doc = (make_resume if kind == "resume" else make_job_doc)(rng, args.split, language, args.long)
            body = (resume_html(doc, layout, language) if kind == "resume"
                    else job_html(doc, layout, language, rng if args.varied else None))

            page.set_content(page_html(body, font, size, accent, tint), wait_until="networkidle")
            page.evaluate("document.fonts.ready")
            name = f"{kind}-{number:03d}"
            image = Image.open(io.BytesIO(page.screenshot(full_page=True))).convert("RGB")
            scanned = rng.random() < SCANNED_SHARE
            if scanned:
                image = scan_effect(image, rng)
            image.save(folder / "images" / f"{name}.jpg", quality=92)
            page.pdf(path=str(folder / "pdf" / f"{name}.pdf"), format="A4", print_background=True)

            records.append({
                "id": name, "language": language, "layout": layout, "font": font, "scanned": scanned,
                "image": f"images/{name}.jpg", "pdf": f"pdf/{name}.pdf",
                "text": page.evaluate("document.body.innerText"), "answer": doc,
            })
            print(f"{number}/{len(kinds)} {name} {language} {layout} {font}{' scanned' if scanned else ''}", flush=True)
        browser.close()

    (folder / "answers.json").write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
