"""Build thai-doc-ocr/results/thaiocrbench_300.md from official eval.py outputs (ollama_res/*.json)
plus runner meta files (ollama_pred/*.meta.jsonl). Scores are the per-item `score` written by eval.py;
per-task mean is computed exactly like eval.py's printout (sum(score)/count)."""
import json
import os
import random
import statistics as st
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
OUT = os.path.abspath(os.path.join(ROOT, "..", "thai-doc-ocr", "results", f"thaiocrbench_{N}.md"))
ROOT = os.environ.get("REPORT_ROOT", ROOT)  # for dry runs only
OUT = os.environ.get("REPORT_OUT", OUT)

MODELS = [("typhoon", "Typhoon OCR 1.5 2B (Q4_K_M)"),
          ("qwen3vl-base-q8", "Qwen3-VL-2B base (q8_0)"),
          ("qwen3vl-thaidoc-q8", "Qwen3-VL-2B thaidoc (q8_0)")]
# grouping = which scoring function eval.py calls for each task
GROUPS = {
    "TED": ["Table parsing", "Chart parsing", "Document parsing"],
    "BMFL": ["Fine-grained text recognition", "Full-page OCR", "Handwritten content extraction"],
    "F1": ["Key information extraction", "Key information mapping"],
    "ANLS": ["Text recognition", "Document classification", "Diagram VQA", "Cognition VQA", "Infographics"],
}
TASKS = [t for g in GROUPS.values() for t in g]
GROUP_OF = {t: g for g, ts in GROUPS.items() for t in ts}
VQA_TASKS = ["Diagram VQA", "Cognition VQA", "Infographics"]


def load(m):
    res = json.load(open(os.path.join(ROOT, "ollama_res", f"{m}_300.json"), encoding="utf-8"))[:N]
    meta = [json.loads(l) for l in open(os.path.join(ROOT, "ollama_pred", f"{m}_300.json.meta.jsonl"), encoding="utf-8")]
    meta = [x for x in meta if x["idx"] < N]
    return res, meta


def task_means(items):
    out = {}
    for t in TASKS:
        s = [float(x["score"]) for x in items if x["type"] == t]
        out[t] = sum(s) / len(s) if s else float("nan")
    return out


def summarize(tm):
    g = {k: st.mean(tm[t] for t in ts) for k, ts in GROUPS.items()}
    return st.mean(tm[t] for t in TASKS), g


def paired_bootstrap(a_items, b_items, B=2000, seed=0):
    """Stratified (by task) paired bootstrap of macro-average difference b - a, plus per-task CIs."""
    rng = random.Random(seed)
    by_task = {t: [i for i, x in enumerate(a_items) if x["type"] == t] for t in TASKS}
    by_task = {t: v for t, v in by_task.items() if v}
    per = {t: [0.0] for t in TASKS}
    diff = [float(b["score"]) - float(a["score"]) for a, b in zip(a_items, b_items)]
    macro = []
    for t in by_task:
        per[t] = []
    for _ in range(B):
        tmeans = []
        for t, idx in by_task.items():
            smp = [diff[rng.choice(idx)] for _ in idx]
            m = sum(smp) / len(smp)
            per[t].append(m)
            tmeans.append(m)
        macro.append(sum(tmeans) / len(tmeans))

    def ci(v):
        v = sorted(v)
        return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]
    return ci(macro), {t: ci(v) for t, v in per.items()}, diff


def clip(s, n=220):
    s = (s or "").replace("\r", "")
    s = s if len(s) <= n else s[:n] + " …"
    return s.replace("```", "ˋˋˋ")


def f(x):
    return f"{x:.3f}"


def main():
    data = {m: load(m) for m, _ in MODELS}
    ids0 = [x["id"] for x in data[MODELS[0][0]][0]]
    for m, _ in MODELS:
        assert [x["id"] for x in data[m][0]] == ids0, f"{m}: different sample set/order"
    n = len(ids0)
    counts = {t: sum(1 for x in data[MODELS[0][0]][0] if x["type"] == t) for t in TASKS}
    tm = {m: task_means(data[m][0]) for m, _ in MODELS}
    summ = {m: summarize(tm[m]) for m, _ in MODELS}
    L = []
    w = L.append

    w(f"# ThaiOCRBench ({n} ข้อ): Typhoon OCR vs Qwen3-VL-2B base vs Qwen3-VL-2B thaidoc\n")
    w("> ใช้เทียบ 3 โมเดลนี้กันเองเท่านั้น ห้ามเทียบตรง ๆ กับตัวเลขในเปเปอร์ (ดูหัวข้อข้อจำกัด)\n")
    w("## วิธีวัด\n")
    w(f"- ชุดข้อ: `scb10x/ThaiOCRBench` split test สุ่มแบบเดียวกับ `--max_samples` ของทางการ (map, `shuffle(seed=42)`, `select(range({n}))`) ทุกโมเดลได้ {n} ข้อชุดเดียวกัน ลำดับเดียวกัน")
    w("- Prompt: ดึง lambda จาก `eval_scripts/run_inference.py` มาใช้ตรง ๆ (`\"Instruction: \" + question + \"\\nตอบเฉพาะข้อความตามภาพตรง ๆ อย่างกระชับ โดยไม่ใส่คำอธิบายเพิ่มเติม\"`) ไม่มี system prompt")
    w("- Inference: Ollama `/api/chat` รูปเป็น base64 PNG, temperature 0, num_predict 4096, num_ctx 8192, seed 42, ย่อรูปให้ด้านยาวไม่เกิน 1,800 px (LANCZOS, ย่ออย่างเดียว ไม่ขยาย)")
    w("- ให้คะแนน: `eval_scripts/eval.py` ของทางการ ไม่ได้แก้ (รันด้วย `PYTHONUTF8=1` เพราะ eval.py เปิดไฟล์โดยไม่ระบุ encoding)")
    w("- Environment ของ eval: Python 3.12 venv แยก, ติดตั้งตาม requirements.txt ยกเว้น Polygon3 (build บน Windows ไม่ได้ และใช้แค่หมวด text spotting ภาษาอังกฤษที่ไม่อยู่ใน 13 หมวด) เพิ่ม pythainlp ที่ eval ใช้แต่ไม่อยู่ใน requirements และ pin `nltk==3.9.1` เพราะ nltk 3.10 เพิ่มเพดาน 2,000 ตัวอักษรใน `edit_distance` ทำให้ eval.py crash กับคำตอบยาว (สูตรคะแนนเหมือนเดิม)")
    w("- ค่าเฉลี่ยรวม = ค่าเฉลี่ยของ 13 หมวด (macro) ค่ากลุ่ม = ค่าเฉลี่ยของหมวดในกลุ่ม แบ่งกลุ่มตามฟังก์ชันที่ eval.py ใช้ให้คะแนน")
    w("  - TED: Table / Chart / Document parsing")
    w("  - BMFL (BLEU, METEOR, F1, 1-edit distance): Fine-grained text recognition, Full-page OCR, Handwritten content extraction")
    w("  - F1: Key information extraction / mapping")
    w("  - ANLS: Text recognition, Document classification, Diagram VQA, Cognition VQA, Infographics\n")

    w("## คะแนนรวม\n")
    w("| โมเดล | เฉลี่ยรวม | TED | BMFL | F1 | ANLS |")
    w("|---|---|---|---|---|---|")
    cols = ["avg"] + list(GROUPS)
    val = {m: {"avg": summ[m][0], **summ[m][1]} for m, _ in MODELS}
    best = {c: max(val[m][c] for m, _ in MODELS) for c in cols}
    for m, name in MODELS:
        cells = [f"**{f(val[m][c])}**" if val[m][c] == best[c] else f(val[m][c]) for c in cols]
        w(f"| {name} | {' | '.join(cells)} |")
    w("\nตัวหนา = สูงสุดในคอลัมน์\n")

    w("## คะแนนรายหมวด\n")
    w("ตัวหนา = สูงสุดในหมวด\n")
    w("| หมวด | กลุ่ม | n | Typhoon | Base | Thaidoc | Thaidoc - Base |")
    w("|---|---|---|---|---|---|---|")
    for t in TASKS:
        vals = [tm[m][t] for m, _ in MODELS]
        best = max(vals)
        cells = [f"**{f(v)}**" if v == best else f(v) for v in vals]
        d = tm["qwen3vl-thaidoc-q8"][t] - tm["qwen3vl-base-q8"][t]
        w(f"| {t} | {GROUP_OF[t]} | {counts[t]} | {' | '.join(cells)} | {d:+.3f} |")
    w("")

    w("## เวลาและข้อที่มีปัญหา\n")
    w("| โมเดล | เวลาเฉลี่ย/ข้อ (s) | มัธยฐาน (s) | รวม (ชม.) | error/timeout | ชนเพดาน 4096 token (วนลูป) | ตอบว่าง (ไม่ error) |")
    w("|---|---|---|---|---|---|---|")
    loops = {}
    for m, name in MODELS:
        res, meta = data[m]
        secs = [x["seconds"] for x in meta]
        err = sum(1 for x in meta if x.get("error"))
        lp = [x["idx"] for x in meta if x.get("done_reason") == "length"]
        loops[m] = lp
        empty = sum(1 for x, mm in zip(res, sorted(meta, key=lambda z: z["idx"])) if not mm.get("error") and not (x["predict"] or "").strip())
        w(f"| {name} | {st.mean(secs):.1f} | {st.median(secs):.1f} | {sum(secs) / 3600:.2f} | {err} | {len(lp)} | {empty} |")
    w("")
    w("วนลูปต่อหมวด (จำนวนข้อที่ generate จนชน 4096 token):\n")
    w("| หมวด | Typhoon | Base | Thaidoc |")
    w("|---|---|---|---|")
    for t in TASKS:
        row = [sum(1 for i in loops[m] if data[m][0][i]["type"] == t) for m, _ in MODELS]
        if any(row):
            w(f"| {t} | {' | '.join(map(str, row))} |")
    meta0 = data[MODELS[0][0]][1]
    resized = sum(1 for x in meta0 if max(x["orig_size"]) > 1800)
    w(f"\nรูปที่ถูกย่อ (ด้านยาวเกิน 1,800 px): {resized} จาก {n} รูป\n")

    # ---- thaidoc vs base
    a, b = data["qwen3vl-base-q8"][0], data["qwen3vl-thaidoc-q8"][0]
    (lo, hi), per_ci, diff = paired_bootstrap(a, b)
    d_all = summ["qwen3vl-thaidoc-q8"][0] - summ["qwen3vl-base-q8"][0]
    wins = sum(1 for d in diff if d > 1e-9)
    loss = sum(1 for d in diff if d < -1e-9)
    w("## Thaidoc ดีกว่า Base ไหม\n")
    w(f"- คะแนนเฉลี่ยรวม: Thaidoc {f(summ['qwen3vl-thaidoc-q8'][0])} vs Base {f(summ['qwen3vl-base-q8'][0])} ต่างกัน **{d_all:+.3f}** (95% CI จาก paired bootstrap แบ่งชั้นตามหมวด: {lo:+.3f} ถึง {hi:+.3f})")
    w(f"- รายข้อ: Thaidoc ดีกว่า {wins} ข้อ, แย่กว่า {loss} ข้อ, เท่ากัน {n - wins - loss} ข้อ")
    for g in GROUPS:
        dg = summ["qwen3vl-thaidoc-q8"][1][g] - summ["qwen3vl-base-q8"][1][g]
        w(f"- กลุ่ม {g}: {dg:+.3f}")
    w("")
    w("| หมวด | n | Base | Thaidoc | ต่าง | 95% CI | สรุป |")
    w("|---|---|---|---|---|---|---|")
    for t in sorted(TASKS, key=lambda t: tm["qwen3vl-thaidoc-q8"][t] - tm["qwen3vl-base-q8"][t], reverse=True):
        d = tm["qwen3vl-thaidoc-q8"][t] - tm["qwen3vl-base-q8"][t]
        clo, chi = per_ci[t]
        if counts[t] < 5:
            verdict = f"มีแค่ {counts[t]} ข้อ สรุปไม่ได้"
        elif clo == chi == 0:
            verdict = "คะแนนเท่ากันทุกข้อ"
        else:
            verdict = "ดีขึ้นชัด" if clo > 0 else ("แย่ลงชัด" if chi < 0 else "แยกจาก noise ไม่ได้")
        w(f"| {t} | {counts[t]} | {f(tm['qwen3vl-base-q8'][t])} | {f(tm['qwen3vl-thaidoc-q8'][t])} | {d:+.3f} | {clo:+.3f} ถึง {chi:+.3f} | {verdict} |")
    w("\n<!-- NARRATIVE -->\n")

    # ---- typhoon VQA note
    ty_low = [t for t in VQA_TASKS if tm["typhoon"][t] == min(tm[m][t] for m, _ in MODELS)]
    if ty_low:
        w("## หมายเหตุเรื่อง Typhoon กับหมวด VQA\n")
        for t in VQA_TASKS:
            rank = 1 + sum(1 for m, _ in MODELS if tm[m][t] > tm["typhoon"][t])
            w(f"- {t}: Typhoon {f(tm['typhoon'][t])} (อันดับ {rank} จาก 3, n={counts[t]})")
        w("- ส่วนที่ต่ำ อาจเป็นเพราะ Typhoon OCR เทรนมากับ prompt เฉพาะของตัวเอง (งาน document parsing ออกมาเป็น markdown/HTML) ไม่ได้เทรนให้ตอบคำถามสั้น ๆ ตาม prompt ทั่วไป README ของ ThaiOCRBench เองก็ไม่ได้วัด Typhoon OCR ด้วยเหตุผลนี้ (\"The response format is different\" และ \"only supports a single task — Document Parsing\")\n")

    # ---- wrong-answer examples
    w("## ตัวอย่างคำตอบที่ผิด (หมวดละ 1 ข้อ ของโมเดลที่คะแนนต่ำสุดในหมวดนั้น)\n")
    w("เลือกข้อที่โมเดลนั้นได้คะแนนต่ำสุดในหมวด ถ้าเท่ากันเลือกข้อที่โมเดลอื่นทำได้ดีกว่า ตัดข้อความให้สั้นลง\n")
    names = dict(MODELS)
    for t in TASKS:
        worst = min((m for m, _ in MODELS), key=lambda m: tm[m][t])
        res = data[worst][0]
        idx = [i for i, x in enumerate(res) if x["type"] == t]
        if not idx:
            continue
        others = [m for m, _ in MODELS if m != worst]
        i = min(idx, key=lambda i: (float(res[i]["score"]),
                                    -max(float(data[o][0][i]["score"]) for o in others)))
        x = res[i]
        q = x["question"].removeprefix("Instruction: ").split("\nตอบเฉพาะข้อความ")[0]
        loop = " (ชน 4096 token)" if i in loops[worst] else ""
        w(f"### {t}: {names[worst]} ได้ {f(float(x['score']))}{loop}\n")
        w(f"- id `{x['id']}`, คะแนนหมวด {f(tm[worst][t])}")
        w(f"- คำถาม: {clip(q, 200)}")
        w("- เฉลย:\n\n```\n" + clip(x["answers"][0]) + "\n```")
        w("- คำตอบโมเดล:\n\n```\n" + (clip(x["predict"]) or "(ว่าง)") + "\n```\n")

    w("## ข้อจำกัด\n")
    small = ", ".join(f"{t} {counts[t]} ข้อ" for t in TASKS if counts[t] < 5)
    w(f"- วัดแค่ {n} จาก 2,808 ข้อ (หมวดละประมาณ {n // 13} ข้อ แต่สุ่มแล้วไม่เท่ากัน ตั้งแต่ {min(counts.values())} ถึง {max(counts.values())} ข้อ) คะแนนรายหมวดจึงแกว่งได้มาก ดูช่วง CI ประกอบ" + (f" หมวดที่มีน้อยมาก: {small} ข้อเดียวก็เปลี่ยนคะแนนหมวดได้หลายสิบจุด" if small else ""))
    w("- CI ในรายงานคิดจากการสุ่มข้อซ้ำ (bootstrap) ภายในชุดนี้ บอกความไม่แน่นอนจากการเลือกข้อ ไม่ได้รวมความแปรปรวนจากการรันซ้ำหรือจาก quantization")
    w("- ใช้โมเดล quantize ผ่าน Ollama (Typhoon Q4_K_M, Qwen q8_0) ไม่ใช่ bf16 ผ่าน vLLM แบบในเปเปอร์ และ Typhoon quantize หนักกว่าอีก 2 ตัว")
    w("- ย่อรูปให้ด้านยาวไม่เกิน 1,800 px และ num_predict 4096 (ทางการใช้ max_tokens 11000) คำตอบยาว ๆ ที่เกิน 4096 token จะถูกตัด")
    w("- รันครั้งเดียว (temperature 0) ไม่ได้วัดความแปรปรวนจากการรันซ้ำ")
    w("- ด้วยเหตุผลข้างบน **เทียบตรง ๆ กับตัวเลขในเปเปอร์ (arXiv 2511.04479) ไม่ได้** ใช้เทียบ 3 โมเดลนี้กันเองเท่านั้น")
    w("- eval.py ของทางการให้คะแนนหมวด ANLS เต็ม 1 ถ้าเฉลยเป็น substring ของคำตอบ คำตอบยาวที่บังเอิญมีเฉลยอยู่จึงได้เต็ม และ Document parsing ที่คำตอบว่างยังได้ประมาณ 0.05 (STEDS) ส่วน Table parsing (TEDS) ให้คะแนนติดลบได้เล็กน้อยเมื่อ HTML ที่ตอบยาวกว่าเฉลยมาก (เจอ 3 ข้อ ต่ำสุด -0.047) รายงานนี้ใช้คะแนนตามที่ eval.py ให้ ไม่ได้ปัดเป็น 0")
    w("- ThaiOCRBench ใช้เป็นชุดทดสอบเท่านั้น ไม่ได้ปนกับ `thai-doc-ocr/data/train`\n")
    w("## ไฟล์ที่เกี่ยวข้อง\n")
    w("- runner: `ThaiOCRBench/ollama_eval/run_ollama.py`, driver: `ollama_eval/run_all.ps1`, รายงาน: `ollama_eval/make_report.py`")
    w(f"- คำตอบดิบ (รูปแบบเดียวกับ run_inference.py): `ThaiOCRBench/ollama_pred/<model>_300.json` (+ `.meta.jsonl` เวลา/token ต่อข้อ) ชื่อไฟล์ยังเป็น _300 เพราะวางแผนไว้ 300 ข้อ แต่ข้างในมี {n} ข้อแรกของลำดับเดียวกัน รันต่อเป็น 300 ได้ด้วย `run_all.ps1`")
    w("- ผลจาก eval.py: `ThaiOCRBench/ollama_res/<model>_300.json` และ `.eval.txt`")

    text = "\n".join(L) + "\n"
    if os.path.exists(OUT):  # keep hand-written narrative if present
        old = open(OUT, encoding="utf-8").read()
        if "<!-- NARRATIVE -->" in old and "<!-- /NARRATIVE -->" in old:
            narr = old.split("<!-- NARRATIVE -->")[1].split("<!-- /NARRATIVE -->")[0]
            text = text.replace("<!-- NARRATIVE -->\n", "<!-- NARRATIVE -->" + narr + "<!-- /NARRATIVE -->\n")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write(text)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
