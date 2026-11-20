"""Measure how well a text model turns a document's text into the JSON the answers hold.

    python evaluate_json.py qwen3-1.7b-q8                    # from Typhoon's reading of each test page
    python evaluate_json.py qwen3-1.7b-q8 --source truth     # from the page's real text: the JSON step alone
    python evaluate_json.py qwen3-1.7b-q8 --split test_long --limit 5

The model gets the document type (the web app knows which side a file was added on) and a JSON schema that Ollama
enforces, so the answer always parses. Per field, values are compared after evaluate.normalize:
  recall     share of the answer's values the model put in the right field, copied exactly
  precision  share of the model's values that are in the answer: low means made-up or misplaced values
Lists (skills, experience...) are compared as multisets per field, so order doesn't matter.
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import ollama

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "web-search-bot"))

import attachments
from evaluate import RESULTS, fields_of, normalize

TEXT = {"type": "string"}
LIST = {"type": "array", "items": TEXT}
RESUME_SCHEMA = {
    "type": "object",
    "properties": {
        "name": TEXT, "position": TEXT, "email": TEXT, "phone": TEXT, "location": TEXT, "summary": TEXT,
        "skills": LIST, "languages": LIST,
        "education": {"type": "array", "items": {"type": "object", "properties": {
            "degree": TEXT, "field": TEXT, "school": TEXT, "year": TEXT},
            "required": ["degree", "field", "school", "year"]}},
        "experience": {"type": "array", "items": {"type": "object", "properties": {
            "title": TEXT, "company": TEXT, "start": TEXT, "end": {"type": ["string", "null"]}, "details": LIST},
            "required": ["title", "company", "start", "end", "details"]}},
    },
    "required": ["name", "position", "email", "phone", "location", "summary", "skills", "languages",
                 "education", "experience"],
}
JOB_SCHEMA = {
    "type": "object",
    "properties": {
        "position": TEXT, "company": TEXT, "location": TEXT, "employment_type": TEXT, "salary": TEXT,
        "requirements": {"type": "object", "properties": {
            "skills": LIST, "abilities": LIST, "personality": LIST, "other": LIST},
            "required": ["skills", "abilities", "personality", "other"]},
        "responsibilities": LIST, "benefits": LIST, "contact": TEXT,
    },
    "required": ["position", "company", "location", "employment_type", "salary", "requirements",
                 "responsibilities", "benefits", "contact"],
}
SCHEMAS = {"resume": RESUME_SCHEMA, "job_description": JOB_SCHEMA}

PROMPT = (
    "Extract the {kind} below into JSON. Copy every value exactly as it is written, in its original language: "
    "do not translate, reword, shorten or add anything. Write years as they appear (Thai documents use the "
    "Buddhist Era). Leave a field empty when the document doesn't have it.{extra}"
)
EXTRA = {
    "resume": " For a job that is still going on, set end to null.",
    "job_description": " Sort each requirement into skills (tools and technical skills), abilities (experience "
                       "and know-how), personality (character and soft skills) or other (degree, language, licence).",
}
OPTIONS = {"temperature": 0, "num_ctx": 8192, "num_predict": 2048}


def to_json(model: str, doc_type: str, text: str, schema: bool = True) -> tuple[dict | None, float]:
    """With schema=True Ollama enforces the fields, but writes keys in alphabetical order ("company" before "title").
    A model trained on the documents' own order then fills them one key off, so for it pass schema=False: any valid
    JSON, keys in the order the model learned."""
    kind = "resume" if doc_type == "resume" else "job description"
    start = time.monotonic()
    reply = ollama.chat(model=model, think=False, format=SCHEMAS[doc_type] if schema else "json", options=OPTIONS, messages=[
        {"role": "system", "content": PROMPT.format(kind=kind, extra=EXTRA[doc_type])},
        {"role": "user", "content": text}])
    try:
        return json.loads(reply["message"]["content"]), time.monotonic() - start
    except json.JSONDecodeError:  # ran out of num_predict mid-answer
        return None, time.monotonic() - start


def compare(truth: dict, predicted: dict | None) -> dict:
    """Per field: [values matched, values in the answer, values the model gave]."""
    want, got = defaultdict(Counter), defaultdict(Counter)
    for name, value in fields_of(truth):
        want[name][normalize(value)] += 1
    for name, value in fields_of(predicted or {}):
        if normalize(value):
            got[name][normalize(value)] += 1
    return {name: [sum((want[name] & got[name]).values()), sum(want[name].values()), sum(got[name].values())]
            for name in set(want) | set(got)}


def totals(rows: list[dict]) -> dict:
    matched = sum(c[0] for r in rows for c in r["fields"].values())
    wanted = sum(c[1] for r in rows for c in r["fields"].values())
    given = sum(c[2] for r in rows for c in r["fields"].values())
    return {"documents": len(rows), "recall": round(matched / max(wanted, 1), 3),
            "precision": round(matched / max(given, 1), 3),
            "valid_json": round(sum(r["valid"] for r in rows) / len(rows), 3),
            "seconds": round(sum(r["seconds"] for r in rows) / len(rows), 1)}


def summarize(rows: list[dict]) -> dict:
    groups = {"all": rows}
    for key in ("doc_type", "language"):
        for value in sorted({r[key] for r in rows}):
            groups[f"{key}={value}"] = [r for r in rows if r[key] == value]
    summary = {name: totals(group) for name, group in groups.items()}
    by_field = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        for name, counts in row["fields"].items():
            by_field[name] = [a + b for a, b in zip(by_field[name], counts)]
    summary["by_field"] = {name: {"recall": round(m / max(w, 1), 3), "precision": round(m / max(g, 1), 3)}
                           for name, (m, w, g) in sorted(by_field.items())}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--split", default="test")
    parser.add_argument("--source", choices=["typhoon", "truth"], default="typhoon",
                        help="typhoon: what Typhoon read from the page (as in the web app); truth: the page's text")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--format", choices=["schema", "json"], default="schema",
                        help="schema: Ollama enforces the fields (alphabetical key order); json: any valid JSON")
    args = parser.parse_args()

    folder = HERE / "data" / args.split
    records = json.loads((folder / "answers.json").read_text(encoding="utf-8"))
    if args.source == "typhoon":
        suffix = "" if args.split == "test" else f"-{args.split}"
        read = json.loads((RESULTS / f"typhoon-whole{suffix}-text.json").read_text(encoding="utf-8"))
        texts = {doc: attachments.clean_transcript(text) for doc, text in read.items()}
    else:
        texts = {r["id"]: r["text"] for r in records}
    if args.limit:
        records = records[:args.limit]

    rows, outputs = [], {}
    for number, record in enumerate(records, 1):
        doc_type = record["answer"]["doc_type"]
        predicted, seconds = to_json(args.model, doc_type, texts[record["id"]], args.format == "schema")
        row = {"id": record["id"], "doc_type": doc_type, "language": record["language"], "valid": predicted is not None,
               "seconds": seconds, "fields": compare(record["answer"], predicted)}
        rows.append(row)
        outputs[record["id"]] = predicted
        t = totals([row])
        print(f"{number}/{len(records)} {record['id']} recall={t['recall']:.2f} precision={t['precision']:.2f} "
              f"{seconds:.0f}s", flush=True)

    summary = summarize(rows)
    name = f"json-{args.model.replace('/', '_').replace(':', '_')}-{args.source}" + ("-free" if args.format == "json" else "") + (
        "" if args.split == "test" else f"-{args.split}")
    (RESULTS / f"{name}.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    (RESULTS / f"{name}-output.json").write_text(json.dumps(outputs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_field"}, ensure_ascii=False, indent=1))
    print(json.dumps(summary["by_field"], ensure_ascii=False))


if __name__ == "__main__":
    main()
