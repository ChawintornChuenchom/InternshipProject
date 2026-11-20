"""Turn a generated split into a LlamaFactory dataset.

    python to_llamafactory.py --split train               # image -> text, for a vision model (thai_doc_ocr_train)
    python to_llamafactory.py --split train --task json   # text -> JSON, for a text model (thai_doc_json_train)

Writes LlamaFactory/data/<name>.json and registers it in dataset_info.json. Prompts are the ones evaluate.py and
evaluate_json.py use, so the model learns exactly the task we measure. Never convert a test split for training.
The json task feeds the page's real text: Typhoon reads these pages almost perfectly, and reading 400 pages with it
would take about an hour.
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
FACTORY_DATA = HERE.parent / "LlamaFactory" / "data"
sys.path.insert(0, str(HERE.parent / "web-search-bot"))

import attachments
from evaluate_json import EXTRA, PROMPT

TAGS = {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}


def ocr_row(record: dict, folder: Path) -> dict:
    # LlamaFactory joins image paths onto its data folder, so store them relative to it.
    image = os.path.relpath(folder / record["image"], FACTORY_DATA).replace("\\", "/")
    return {
        "messages": [
            {"role": "user", "content": "<image>" + attachments.TRANSCRIBE_PROMPT},
            {"role": "assistant", "content": record["text"].strip()},
        ],
        "images": [image],
    }


def json_row(record: dict) -> dict:
    doc_type = record["answer"]["doc_type"]
    kind = "resume" if doc_type == "resume" else "job description"
    # doc_type is known, not extracted. Keys sorted: given a schema, Ollama writes keys in alphabetical order
    # ("company" before "title"), and a model trained on the document's order then filled every field one key off.
    answer = {key: value for key, value in record["answer"].items() if key != "doc_type"}
    return {
        "system": PROMPT.format(kind=kind, extra=EXTRA[doc_type]),
        "messages": [
            {"role": "user", "content": record["text"].strip()},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False, sort_keys=True)},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--task", choices=["ocr", "json"], default="ocr")
    args = parser.parse_args()
    if args.split.startswith("test"):  # test and test_long
        sys.exit("The test split is for measuring only; training on it makes the results meaningless.")

    folder = HERE / "data" / args.split
    records = json.loads((folder / "answers.json").read_text(encoding="utf-8"))
    if args.task == "ocr":
        name, rows = f"thai_doc_ocr_{args.split}", [ocr_row(r, folder) for r in records]
        entry = {"columns": {"messages": "messages", "images": "images"}}
    else:
        name, rows = f"thai_doc_json_{args.split}", [json_row(r) for r in records]
        entry = {"columns": {"messages": "messages", "system": "system"}}
    (FACTORY_DATA / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    info_path = FACTORY_DATA / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info[name] = {"file_name": f"{name}.json", "formatting": "sharegpt", **entry, "tags": TAGS}
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(rows)} examples written to {name}.json and registered as '{name}'")


if __name__ == "__main__":
    main()
