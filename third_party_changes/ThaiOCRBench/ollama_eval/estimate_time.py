"""Rough runtime estimate for the 300-sample subset, from gold-answer length per task.
Calibrated on the 5-item smoke test meta files: seconds ~= overhead + tokens / tok_per_s,
tokens ~= answer_chars * tokens_per_char."""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_ollama import build_dataset  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
meta_dir = os.path.join(os.path.dirname(__file__), "..", "ollama_pred", "test5")

ds = build_dataset(N).remove_columns(["image"])
ans_len = [len(a) for a in ds["answer"]]
first5 = ans_len[:5]

for m in ["typhoon", "qwen3vl-base-q8", "qwen3vl-thaidoc-q8"]:
    p = os.path.join(meta_dir, f"{m}.json.meta.jsonl")
    if not os.path.exists(p):
        continue
    meta = [json.loads(l) for l in open(p, encoding="utf-8")]
    toks = [x["eval_count"] or 0 for x in meta]
    secs = [x["seconds"] for x in meta]
    # tokens/char from the long items, decode rate from items with >200 tokens, overhead from short ones
    tpc = sum(toks) / max(1, sum(first5[: len(meta)]))
    long = [(t, s) for t, s in zip(toks, secs) if t > 200]
    short = [s for t, s in zip(toks, secs) if t <= 200]
    ovh = sorted(short)[len(short) // 2] if short else 8
    rate = sum(t for t, _ in long) / max(1e-9, sum(s - ovh for _, s in long)) if long else 30
    est = 0
    per_task = collections.defaultdict(float)
    for task, L in zip(ds["Task"], ans_len):
        t = min(4096, L * tpc)
        s = ovh + t / rate
        est += s
        per_task[task] += s
    print(f"{m:<20} tok/char={tpc:.2f} overhead={ovh:.1f}s decode={rate:.1f} tok/s -> ~{est / 3600:.1f} h for {N}")
    for k, v in sorted(per_task.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<32} {v / 60:6.1f} min")
print("answer chars, first 5:", first5)
print("task counts:", collections.Counter(ds["Task"]))
