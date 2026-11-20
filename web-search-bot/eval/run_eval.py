"""Score models on the hand-labelled cases in cases.json, using the same code the app runs.

    python eval/run_eval.py paibot qwen3:4b-instruct --runs 3

Keep these cases out of any training data, or the scores stop meaning anything.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

import matcher

RESULTS = HERE / "results"
# What each status means for "does the resume show this?". "review" is the app saying a person should decide.
VERDICTS = {"met": "has", "partial": "has", "missing": "no", "review": "unsure"}


def run_checks(cases: dict) -> list[dict]:
    rows = []
    for case in cases["checks"]:
        start = time.monotonic()
        result = matcher.check_requirement(case["requirement"], cases["resumes"][case["resume"]])
        rows.append({**case, "got": VERDICTS[result["status"]], "seconds": time.monotonic() - start})
    return rows


def run_categories(cases: dict) -> int:
    return sum(matcher.category(case["requirement"]) == case["answer"] for case in cases["categories"])


def percent(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "-"


def summarize(runs: list[list[dict]]) -> dict:
    rows = [row for run in runs for row in run]
    has = [r for r in rows if r["answer"] == "has"]
    no = [r for r in rows if r["answer"] == "no"]
    thai = [r for r in rows if r["resume"].startswith("th_")]
    english = [r for r in rows if not r["resume"].startswith("th_")]
    right = lambda group: sum(r["got"] == r["answer"] for r in group)
    same = sum(len({run[i]["got"] for run in runs}) == 1 for i in range(len(runs[0])))
    return {
        "correct": percent(right(rows), len(rows)),
        "wrongly says has": percent(sum(r["got"] == "has" for r in no), len(no)),
        "wrongly says no": percent(sum(r["got"] == "no" for r in has), len(has)),
        "sent to review": percent(sum(r["got"] == "unsure" for r in rows), len(rows)),
        "correct, Thai resumes": percent(right(thai), len(thai)),
        "correct, English resumes": percent(right(english), len(english)),
        **{f"correct, {kind}": percent(right(group), len(group)) for kind, group in by_kind(rows).items()},
        "same answer every run": percent(same, len(runs[0])),
        "seconds per check": f"{sum(r['seconds'] for r in rows) / len(rows):.1f}",
    }


def by_kind(rows: list[dict]) -> dict:
    kinds = {}
    for row in rows:
        kinds.setdefault(row["kind"], []).append(row)
    return dict(sorted(kinds.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    cases = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    RESULTS.mkdir(exist_ok=True)
    table = {}
    for model in args.models:
        matcher.MODEL = model
        matcher.check_requirement("warm up", "warm up")  # loading the model shouldn't count as checking time
        runs = []
        for number in range(args.runs):
            runs.append(run_checks(cases))
            print(f"{model}: run {number + 1} of {args.runs} done", flush=True)
        table[model] = summarize(runs)
        categories = run_categories(cases)
        table[model]["categories right"] = f"{categories}/{len(cases['categories'])}"

        # Everything it got wrong at least once, for reading through by hand.
        misses = Counter()
        for run in runs:
            misses.update((r["resume"], r["requirement"], r["answer"], r["got"]) for r in run if r["got"] != r["answer"])
        lines = [f"{count}/{args.runs}  {resume}  want={want}  got={got}  {text}"
                 for (resume, text, want, got), count in misses.most_common()]
        name = model.replace("/", "_").replace(":", "_")
        (RESULTS / f"{name}-misses.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (RESULTS / "summary.json").write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    for metric in next(iter(table.values())):
        print(f"{metric:28}" + "".join(f"{table[m][metric]:>26}" for m in args.models))


if __name__ == "__main__":
    main()
