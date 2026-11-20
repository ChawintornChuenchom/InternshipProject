"""Compare a resume with a job description, one requirement at a time.

Asked for one overall score, the 1.7B model gave the same resume 75 or 85 depending only on which text came
first, and both were far too high for someone missing 3 of 7 requirements. So the model only does small jobs
(list the requirements, find the line that proves one), and the code checks the proof and does the scoring.
"""

import html
import json
import re

import ollama

MODEL = "paibot"       # any model from `ollama list`
RESUME_CHARS = 6000
JD_CHARS = 4000
MAX_REQUIREMENTS = 12
# A resume plus a job description can pass the Modelfile's num_ctx 4096 once Thai text is tokenized.
# Every call uses the same options: a different num_ctx makes Ollama reload the model.
OPTIONS = {"temperature": 0, "num_ctx": 8192, "num_predict": 1024}

EXTRACT_PROMPT = (
    "List every separate requirement a candidate must meet in the job description: skills, experience, "
    "education, languages. Copy each one in its original language. "
    "Leave out company information, job duties, salary and benefits."
)
EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {"requirements": {"type": "array", "items": {"type": "string"}}},
    "required": ["requirements"],
}

# A job description uploaded as a resume scored 8.2/10: every quote is really in the file, so nothing else
# stops it. Keep "resume" first; listing it last, the model called 7 of 12 real resumes job postings.
DOCUMENT_PROMPT = (
    "What kind of document is this? resume: one person describing their own work history, education and skills. "
    "job: a company describing a job it wants to fill, what applicants need, or how to apply. other: anything else."
)
DOCUMENT_SCHEMA = {
    "type": "object",
    "properties": {"kind": {"type": "string", "enum": ["resume", "job", "other"]}},
    "required": ["kind"],
}
DOCUMENT_CHARS = 3000
# The model calls a real resume with a job description pasted on the end (say, in white text) a resume,
# so this is checked by comparing the text.
PHRASE_CHARS = 12
COPIED_SHARE = 0.5

POSITION_PROMPT = "Give the job title this job description is hiring for, copied in its original language."
POSITION_SCHEMA = {"type": "object", "properties": {"position": {"type": "string"}}, "required": ["position"]}

# One requirement at a time: sorting the whole job description in one go put the same item in several
# categories. The model leans towards whichever category comes first: with personality first, Transformer
# architectures and MLOps became personality traits. With other first they pile up there instead, but that's
# the one category with a closed vocabulary, so OTHER_WORDS can send the strays on to ability.
CATEGORY_PROMPT = (
    "Put one job requirement in the first category that fits.\n"
    "other: a degree or education, a spoken language such as English or Thai, a license, a location or travel, working hours.\n"
    "skill: names a tool, technology, software, framework, cloud service or programming language.\n"
    "personality: only character and soft skills, such as teamwork, responsibility, curiosity, patience, "
    "communication, problem solving. Never a technical subject.\n"
    "ability: technical or professional know-how, a work practice, or something the person must have done."
)
CATEGORY_SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string", "enum": ["personality", "other", "skill", "ability"]}},
    "required": ["category"],
}
OTHER_WORDS = re.compile(
    r"degree|bachelor|master|phd|diploma|graduate|education|english|thai|chinese|japanese|language|toeic|ielts"
    r"|licen[sc]e|office|location|on-?site|remote|hybrid|travel|relocat|hours|shift|weekend"
    r"|ปริญญา|จบ|วุฒิ|การศึกษา|ภาษา|ใบขับขี่|ใบอนุญาต|สถานที่|เดินทาง|ต่างจังหวัด|ทำงานเป็นกะ|เวลาทำงาน",
    re.I,
)

# The topic is what the page uses to word its one-line verdict on a candidate. Left to write that line
# itself, the model swapped strengths with gaps and praised a chef's "excellent English". Asked for the
# noun, it gives "Python"; asked to shorten the requirement, it mostly hands the requirement back.
TOPIC_PROMPT = (
    "What noun is this job requirement about? Reply with the noun phrase only, at most three words, "
    "in the requirement's own language. Do not include years or levels."
)
TOPIC_SCHEMA = {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}
TOPIC_MAX_CHARS = 30  # longer than this and the model repeated the requirement instead of naming it

# Evidence comes before status in the schema so the model has to find the line first and decide after.
CHECK_PROMPT = (
    "You check one job requirement against a resume. First copy the exact line from the resume that is "
    "the best evidence, or an empty string if there is none. Then decide: met, partial or missing."
)
CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {"type": "string"},
        "status": {"type": "string", "enum": ["met", "partial", "missing"]},
    },
    "required": ["evidence", "status"],
}

# Sometimes the quote is real but off topic, like "Head Chef, Riverside Hotel" as proof of good English.
# Seeing only the requirement and the quote, and naming both topics first, the model caught these.
# A plain yes/no question rejected more good quotes, like Thai evidence for an English requirement.
RELEVANCE_PROMPT = (
    "First name the topic of the requirement and the topic of the resume line. "
    "Reply related=true only if the line is about the same topic as the requirement."
)
RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "requirement_topic": {"type": "string"},
        "line_topic": {"type": "string"},
        "related": {"type": "boolean"},
    },
    "required": ["requirement_topic", "line_topic", "related"],
}


def _ask(system: str, user: str, schema: dict) -> dict:
    reply = ollama.chat(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        format=schema,
        think=False,
        options=OPTIONS,
    )
    return json.loads(reply["message"]["content"])


def extract_requirements(jd: str) -> list[str]:
    items = _ask(EXTRACT_PROMPT, f"Job description:\n{jd[:JD_CHARS]}", EXTRACT_SCHEMA)["requirements"]
    unique = {item.strip(): None for item in items if item.strip()}  # dict keeps order, drops repeats
    return list(unique)[:MAX_REQUIREMENTS]


def interpret_jd(jd: str) -> dict:
    """The job title, and each requirement with its topic and category."""
    requirements = extract_requirements(jd)
    position = _ask(POSITION_PROMPT, f"Job description:\n{jd[:JD_CHARS]}", POSITION_SCHEMA)["position"].strip()
    items = [{"requirement": r, "topic": _topic(r), "category": category(r)} for r in requirements]
    return {"position": position, "items": items}


def category(requirement: str) -> str:
    answer = _ask(CATEGORY_PROMPT, f"Requirement: {requirement}", CATEGORY_SCHEMA)["category"]
    return "ability" if answer == "other" and not OTHER_WORDS.search(requirement) else answer


def _topic(requirement: str) -> str:
    topic = _ask(TOPIC_PROMPT, f"Requirement: {requirement}", TOPIC_SCHEMA)["topic"].strip()
    # The model lowercases words; use the spelling in the requirement when the topic comes from it.
    match = re.search(re.escape(topic), requirement, re.I)
    topic = match.group() if match else topic
    return topic if 0 < len(topic) <= TOPIC_MAX_CHARS else requirement


# The checker often glues a heading and several bullets into one "line" ("Soft Skills: Analytical thinking &
# problem-solving, Strong logical ...") although each sits on its own line in the resume. On five real resumes
# this sank every met and partial answer. So a quote is checked piece by piece, split where lines and list items
# are joined; every piece must still be in the resume, so a made-up line gets no credit.
QUOTE_PIECES = re.compile(r"\n|[,;|•]|:\s|\s[-–—]\s")
WORD_GAP = r"[\s*_]+"  # spaces, and Markdown emphasis if some is left in the text


def find_quotes(resume: str, evidence: str) -> list[list[int]]:
    """Where the evidence sits in the resume as [start, end] pairs; empty if any part of it isn't there.

    Whole lines first, so a quote that is really in the resume keeps its place: matched piece by piece,
    "Head Chef, Riverside Hotel" put "Head Chef" on the title line and the relevance check got the wrong text.
    """
    evidence = html.unescape(evidence)
    return _find_parts(resume, evidence.splitlines()) or _find_parts(resume, QUOTE_PIECES.split(evidence))


def _find_parts(resume: str, parts: list[str]) -> list[list[int]]:
    spans = []
    for part in parts:
        words = part.replace("*", "").strip(" \t-•\"'“”.").split()
        if not words:
            continue
        match = re.search(WORD_GAP.join(map(re.escape, words)), resume, re.I)
        if match is None:
            return []
        spans.append([match.start(), match.end()])
    return spans


QUOTE_GAP = 200  # pieces this close together are shown to the relevance check as one stretch of the resume


def quoted_text(resume: str, spans: list[list[int]]) -> str:
    """The quote as the resume has it. Pieces glued without their commas and dashes ("Head Chef Riverside Hotel
    2015 2025") read as a different line to the relevance check, which then called it related to good English."""
    start, end = min(s for s, _ in spans), max(e for _, e in spans)
    if end - start <= QUOTE_GAP + sum(e - s for s, e in spans):
        return resume[start:end]
    return " … ".join(resume[s:e] for s, e in spans)


def _phrases(text: str) -> set[str]:
    squashed = "".join(text.casefold().split())  # Thai has no spaces, so compare runs of characters, not words
    return {squashed[i:i + PHRASE_CHARS] for i in range(len(squashed) - PHRASE_CHARS + 1)}


def document_problem(resume: str, jd: str) -> str:
    """Why this text shouldn't be scored as a resume, shown to the user as is; "" when it's fine."""
    job = _phrases(jd)
    if job and len(job & _phrases(resume)) / len(job) > COPIED_SHARE:
        return "ไฟล์นี้มีข้อความเดียวกับ job description เกินครึ่ง"
    try:
        kind = _ask(DOCUMENT_PROMPT, f"Document:\n{resume[:DOCUMENT_CHARS]}", DOCUMENT_SCHEMA)["kind"]
    except json.JSONDecodeError:
        return ""  # can't tell, and turning away a real resume is the worse mistake
    return {"job": "ไฟล์นี้ดูเป็นประกาศรับสมัครงาน ไม่ใช่เรซูเม่", "other": "ไฟล์นี้ดูไม่ใช่เรซูเม่"}.get(kind, "")


def check_requirement(requirement: str, resume: str) -> dict:
    """Returns the status (met, partial, review or missing) and where the proof sits in the resume."""
    result = {"requirement": requirement, "status": "missing", "spans": []}
    try:
        reply = _ask(CHECK_PROMPT, f"Resume:\n{resume[:RESUME_CHARS]}\n\nRequirement: {requirement}", CHECK_SCHEMA)
        spans = find_quotes(resume, reply["evidence"])
        # The model says "met" with evidence like "Partial" or a line it made up; no real quote, no credit.
        if reply["status"] == "missing" or not spans:
            return result
        quote = quoted_text(resume, spans)
        relevance = _ask(RELEVANCE_PROMPT, f"Requirement: {requirement}\nResume line: {quote}", RELEVANCE_SCHEMA)
    except json.JSONDecodeError:  # ran out of num_predict mid-answer, so nothing can be said about this one
        return {**result, "status": "review"}
    # Turned down by the relevance check: that check also rejects some good quotes (B.Eng. Computer Engineering
    # for a CS degree), so it becomes "review" and a person decides on the page, which does the scoring too.
    return {**result, "status": reply["status"] if relevance["related"] else "review", "spans": spans}
