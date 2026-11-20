"""Measure how well a vision model reads the test documents.

    python evaluate.py qwen3-vl:2b-instruct --mode strips
    python evaluate.py qwen3-vl:2b-instruct --mode whole --limit 12

Two numbers per document: CER (share of characters read wrong, lower is better) and how many of the answer's
fields appear in the text it read. CER punishes reading a two-column page in a different order, which isn't really
a mistake, so read both. Keep the test split out of any training data.
"""

import argparse
import base64
import io
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import httpx
import ollama
from PIL import Image
from rapidfuzz.distance import Levenshtein

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "web-search-bot"))

import attachments

RESULTS = HERE / "results"
NUM_PREDICT = {"strips": 1024, "whole": 4096}


def normalize(text: str) -> str:
    """Compare characters, not formatting: no spaces, Markdown or HTML, one spelling of สระอำ, one case."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = unicodedata.normalize("NFC", text).replace("ํา", "ำ").casefold()
    return re.sub(r"[\s*#|_`•·\-–—]+", "", text)


def fields_of(answer: dict) -> list[tuple[str, str]]:
    """Every value in the JSON answer as (field name, value), with list positions dropped from the name."""
    found = []

    def walk(value, path):
        if isinstance(value, dict):
            for key, inner in value.items():
                walk(inner, f"{path}.{key}" if path else key)
        elif isinstance(value, list):
            for inner in value:
                walk(inner, path)
        elif isinstance(value, str) and path != "doc_type":
            found.append((path, value))

    walk(answer, "")
    return found


API = None  # set by --api: an OpenAI-style server such as `llamafactory-cli api`, used instead of Ollama
API_MAX_SIDE = 1800  # Typhoon OCR is trained on pages with a 1800 px longest side; every model gets the same image


def fit(data: str, max_side: int) -> str:
    image = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode()


def read(model: str, prompt: str, data: str, num_predict: int) -> tuple[str, bool]:
    if API:
        data = fit(data, API_MAX_SIDE)
        reply = httpx.post(f"{API}/v1/chat/completions", timeout=900, json={
            "model": model, "temperature": 0, "max_tokens": num_predict,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}},
                {"type": "text", "text": prompt}]}]})
        reply.raise_for_status()
        choice = reply.json()["choices"][0]
        return attachments.clean_transcript(choice["message"]["content"]), choice["finish_reason"] == "stop"
    reply = ollama.chat(model=model, think=False, options={"temperature": 0, "num_ctx": 8192, "num_predict": num_predict},
                        messages=[{"role": "user", "content": prompt, "images": [data]}])
    return attachments.clean_transcript(reply["message"]["content"]), reply.get("done_reason") == "stop"


def read_document(model: str, path: Path, mode: str) -> tuple[str, bool]:
    prompt = attachments.prompt_for(model)
    data = base64.b64encode(path.read_bytes()).decode()
    parts = attachments.strips(data) if mode == "strips" else [data]
    results = [read(model, prompt, part, NUM_PREDICT[mode]) for part in parts]
    return attachments.join_strips([text for text, _ in results]), all(finished for _, finished in results)


def score(record: dict, text: str) -> dict:
    truth, got = normalize(record["text"]), normalize(text)
    fields = fields_of(record["answer"])
    hits = defaultdict(lambda: [0, 0])
    for name, value in fields:
        hits[name][1] += 1
        hits[name][0] += normalize(value) in got
    return {"cer": Levenshtein.distance(truth, got) / len(truth), "fields": dict(hits)}


def average(rows: list[dict]) -> dict:
    right = sum(c[0] for r in rows for c in r["fields"].values())
    total = sum(c[1] for r in rows for c in r["fields"].values())
    return {"documents": len(rows), "cer": round(sum(r["cer"] for r in rows) / len(rows), 3),
            "fields_found": round(right / total, 3), "finished": round(sum(r["finished"] for r in rows) / len(rows), 3),
            "seconds": round(sum(r["seconds"] for r in rows) / len(rows), 1)}


def summarize(rows: list[dict]) -> dict:
    groups = {"all": rows}
    for key in ("doc_type", "language", "layout", "scanned"):
        for value in sorted({str(r[key]) for r in rows}):
            groups[f"{key}={value}"] = [r for r in rows if str(r[key]) == value]
    summary = {name: average(group) for name, group in groups.items()}

    by_field = defaultdict(lambda: [0, 0])
    for row in rows:
        for name, (right, total) in row["fields"].items():
            by_field[name][0] += right
            by_field[name][1] += total
    summary["by_field"] = {name: round(r / t, 3) for name, (r, t) in sorted(by_field.items())}
    return summary


def balanced(records: list[dict], limit: int) -> list[dict]:
    """Spread a short run over resumes and job descriptions instead of taking only the first file type."""
    resumes = [r for r in records if r["answer"]["doc_type"] == "resume"]
    jobs = [r for r in records if r["answer"]["doc_type"] != "resume"]
    take_jobs = min(len(jobs), max(1, limit * len(jobs) // len(records)))
    return resumes[:limit - take_jobs] + jobs[:take_jobs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--mode", choices=["strips", "whole"], default="strips")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--api", help="URL of an OpenAI-style server, e.g. http://127.0.0.1:8000; model is then only a label")
    parser.add_argument("--max-tokens", type=int, help="cap each read; the longest page is ~410 tokens, so a loop ends sooner")
    args = parser.parse_args()
    global API
    API = args.api and args.api.rstrip("/")
    if args.max_tokens:
        NUM_PREDICT[args.mode] = args.max_tokens

    folder = HERE / "data" / args.split
    records = json.loads((folder / "answers.json").read_text(encoding="utf-8"))
    if args.limit:
        records = balanced(records, args.limit)

    RESULTS.mkdir(exist_ok=True)
    name = f"{args.model.replace('/', '_').replace(':', '_')}-{args.mode}" + ("" if args.split == "test" else f"-{args.split}")
    rows, outputs = [], {}
    for number, record in enumerate(records, 1):
        start = time.monotonic()
        text, finished = read_document(args.model, folder / record["image"], args.mode)
        row = {"id": record["id"], "doc_type": record["answer"]["doc_type"], "language": record["language"],
               "layout": record["layout"], "scanned": record["scanned"], "finished": finished,
               "seconds": time.monotonic() - start, **score(record, text)}
        rows.append(row)
        outputs[record["id"]] = text
        print(f"{number}/{len(records)} {record['id']} cer={row['cer']:.3f} finished={finished} {row['seconds']:.0f}s", flush=True)

    summary = summarize(rows)
    (RESULTS / f"{name}.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    (RESULTS / f"{name}-text.json").write_text(json.dumps(outputs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
