"""ThaiOCRBench inference runner via the Ollama API (Windows-friendly replacement for vLLM).

- Prompt transform is pulled from eval_scripts/run_inference.py at runtime (AST), not retyped.
- Sampling mirrors --max_samples of the official script: map -> shuffle(seed=42) -> select(range(n)).
- Output file has the same format as run_inference.py: list of
  {"type", "id", "question", "answers", "predict"}  -> score with the official eval.py unchanged.
- Per-item stats (time, tokens, done_reason, errors) go to a separate <output>.meta.jsonl.
- Saves after every item; re-running resumes from where it stopped.
"""
import argparse
import ast
import base64
import json
import os
import sys
import time
from io import BytesIO

import requests
from datasets import load_dataset
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OFFICIAL_INFER = os.path.join(HERE, "..", "eval_scripts", "run_inference.py")


def load_official_map_fn(path=OFFICIAL_INFER):
    """Find `dataset.map(lambda x: {...})` in the official script and compile that exact lambda."""
    with open(path, encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "map" and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "dataset" and node.args
                and isinstance(node.args[0], ast.Lambda)):
            lam = node.args[0]
            fn = eval(compile(ast.Expression(body=lam), path, "eval"), {})
            return fn, ast.get_source_segment(src, lam)
    raise RuntimeError("Could not find dataset.map(lambda ...) in official run_inference.py")


def build_dataset(max_samples):
    map_fn, _ = load_official_map_fn()
    ds = load_dataset("scb10x/ThaiOCRBench")["test"]
    ds = ds.map(map_fn)
    # identical to official: if args.max_samples: shuffle(seed=42) then select
    if max_samples:
        ds = ds.shuffle(seed=42)
        ds = ds.select(range(min(len(ds), max_samples)))
    return ds


def image_to_b64(img, max_side):
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii"), (w, h), img.size


def atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_meta(path):
    meta = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    m = json.loads(line)
                    meta[m["idx"]] = m
    return meta


def write_meta(path, meta):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for idx in sorted(meta):
            f.write(json.dumps(meta[idx], ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def fmt_hms(sec):
    sec = int(sec)
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Ollama model name")
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--max_samples", type=int, default=300)
    ap.add_argument("--max_side", type=int, default=1800)
    ap.add_argument("--num_predict", type=int, default=4096)
    ap.add_argument("--num_ctx", type=int, default=8192)
    ap.add_argument("--timeout", type=int, default=600, help="seconds per request")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--retry_errors", action="store_true", help="redo items that errored/timed out")
    ap.add_argument("--print_prompt", action="store_true")
    args = ap.parse_args()

    if os.name == "nt":  # keep Windows awake while this process runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

    _, lam_src = load_official_map_fn()
    print(f"[prompt] using official map lambda from run_inference.py:\n{lam_src}\n", flush=True)

    ds = build_dataset(args.max_samples)
    n = len(ds)
    ids = list(ds["Id"])
    print(f"[data] {n} samples, first ids: {ids[:5]}", flush=True)
    if args.print_prompt:
        print(repr(ds[0]["question"]), flush=True)

    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"sample_ids_{n}.json"), "w", encoding="utf-8") as f:
        json.dump(ids, f)

    meta_path = args.output_path + ".meta.jsonl"
    preds = []
    if os.path.exists(args.output_path):
        with open(args.output_path, encoding="utf-8") as f:
            preds = json.load(f)
        if [p["id"] for p in preds] != ids[:len(preds)]:
            sys.exit("[resume] existing output does not match this sample order; aborting (move the file away)")
    meta = {k: v for k, v in load_meta(meta_path).items() if k < len(preds)}
    todo = [i for i in range(n) if i >= len(preds) or (args.retry_errors and meta.get(i, {}).get("error"))]
    print(f"[resume] {len(preds)} already saved, {len(todo)} to run", flush=True)

    url = args.host.rstrip("/") + "/api/chat"
    t_run, done_now = time.time(), 0
    for i in todo:
        data = ds[i]
        b64, orig_size, sent_size = image_to_b64(data["image"], args.max_side)
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": data["question"], "images": [b64]}],
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0, "num_predict": args.num_predict,
                        "num_ctx": args.num_ctx, "seed": 42},
        }
        for attempt in range(6):  # retry only when the Ollama server itself is unreachable
            err, predict, resp = None, "", {}
            t0 = time.time()
            try:
                r = requests.post(url, json=payload, timeout=args.timeout)
                if r.status_code != 200:
                    err = f"HTTP {r.status_code}: {r.text[:300]}"
                else:
                    resp = r.json()
                    predict = resp.get("message", {}).get("content", "") or ""
            except requests.exceptions.Timeout:
                err = f"timeout>{args.timeout}s"
            except requests.exceptions.ConnectionError as e:
                err = f"ConnectionError: {e}"
                print(f"[warn] Ollama unreachable (attempt {attempt + 1}), waiting 30s", flush=True)
                time.sleep(30)
                continue
            except Exception as e:  # JSON error, ...
                err = f"{type(e).__name__}: {e}"
            break
        dt = time.time() - t0
        if err:
            predict = ""

        rec = {"type": data["Task"], "id": data["Id"], "question": data["question"],
               "answers": [data["answer"]], "predict": predict}
        if i < len(preds):
            preds[i] = rec
        else:
            preds.append(rec)
        meta[i] = {"idx": i, "id": data["Id"], "type": data["Task"], "seconds": round(dt, 2),
                   "error": err, "done_reason": resp.get("done_reason"),
                   "eval_count": resp.get("eval_count"), "prompt_eval_count": resp.get("prompt_eval_count"),
                   "load_seconds": round(resp.get("load_duration", 0) / 1e9, 2),
                   "orig_size": list(orig_size), "sent_size": list(sent_size)}
        atomic_write_json(args.output_path, preds)
        write_meta(meta_path, meta)

        done_now += 1
        avg = (time.time() - t_run) / done_now
        left = len(todo) - done_now
        flag = "ERROR " + err if err else (resp.get("done_reason") or "")
        print(f"[{i + 1:>3}/{n}] {data['Task'][:30]:<30} {dt:6.1f}s tok={resp.get('eval_count')!s:>5} "
              f"{flag} | avg {avg:.1f}s/item ETA {fmt_hms(avg * left)}", flush=True)

    n_err = sum(1 for m in meta.values() if m.get("error"))
    n_len = sum(1 for m in meta.values() if m.get("done_reason") == "length")
    secs = [m["seconds"] for m in meta.values()]
    print(f"[done] {len(preds)} saved | errors/timeouts={n_err} | hit num_predict={n_len} | "
          f"mean {sum(secs) / max(1, len(secs)):.1f}s/item", flush=True)


if __name__ == "__main__":
    main()
