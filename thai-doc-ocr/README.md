# Thai Doc OCR

ชุดทดสอบการอ่านเรซูเม่และ Job Description ภาษาไทยด้วยโมเดล ใช้วัดผลก่อนและหลัง fine-tune

เอกสารทั้งหมดเป็นข้อมูลสมมติ ไม่มีข้อมูลของคนจริง

## วิธีใช้

รันในโฟลเดอร์นี้ โดยใช้ `.venv` ของ `NewProject` และเปิด Ollama ค้างไว้

```powershell
# สร้างชุดทดสอบ 50 ฉบับ (สร้างซ้ำแล้วได้เอกสารเดิม)
..\.venv\Scripts\python.exe generate.py --split test --resumes 36 --jobs 14

# สร้างชุดทดสอบเอกสารยาว 20 ฉบับใน data/test_long (งาน 4-5 ที่ ปริญญาตรีและโท ยาวประมาณ 1,300 ตัวอักษร)
# ชุดเดิมยังสร้างซ้ำได้เหมือนเดิมทุกตัว
..\.venv\Scripts\python.exe generate.py --split test --long --resumes 15 --jobs 5

# สร้างข้อมูลเทรน (ชื่อคน บริษัท และมหาวิทยาลัยไม่ซ้ำกับชุดทดสอบ)
..\.venv\Scripts\python.exe generate.py --split train --resumes 400 --jobs 100

# วัดผลโมเดล แบบตัด 3 แถบ หรืออ่านทั้งหน้า
..\.venv\Scripts\python.exe evaluate.py qwen3-vl:2b-instruct --mode strips
..\.venv\Scripts\python.exe evaluate.py qwen3-vl:2b-instruct --mode whole --limit 12
```

ผลอยู่ใน `results/` ส่วน `<ชื่อโมเดล>-text.json` คือข้อความที่โมเดลอ่านได้ ใช้ดูตัวอย่างที่อ่านผิด

## เทรนด้วย LlamaFactory

Qwen3-VL-2B QLoRA 4-bit เทรนบน RTX 2050 4GB ได้ แต่ต้องเทรนผ่าน `train.py` แทน `llamafactory-cli train` เพราะต้องแก้ 2 เรื่องก่อน

- **คิด loss เฉพาะคำตอบ** (`answer_only_loss.py`): ปกติโมเดลคิดคะแนนคำศัพท์ 151,936 คำทุกตำแหน่ง รวมตำแหน่งของรูป ใช้หน่วยความจำประมาณ 1GB ส่วนนี้คิดเฉพาะส่วนคำตอบ ทีละ 128 ตำแหน่ง ค่า loss เท่าเดิม
- **จำกัดหน่วยความจำของ PyTorch ที่ 3GB** (`train.py`): Windows ไม่แจ้ง out of memory แต่แอบยืม RAM ทำให้ช้าและล้มด้วย CUBLAS error การจำกัดหน่วยความจำทำให้ PyTorch คืน cache ก่อนเกิน

ผลบนการ์ดนี้: รูปประมาณ 1,700 token ใช้หน่วยความจำประมาณ 2,600MB ขั้นละ 46 วินาที ข้อมูล 400 ฉบับ 2 epochs ใช้เวลา 1 ชั่วโมง 15 นาที

```powershell
# รันในโฟลเดอร์ LlamaFactory
# แปลงข้อมูลเทรนเป็นรูปแบบของ LlamaFactory และลงทะเบียนใน data/dataset_info.json
..\.venv\Scripts\python.exe ..\thai-doc-ocr\to_llamafactory.py --split train

# เทรน แล้วรวม adapter เข้ากับโมเดล (รวมบน CPU)
.venv\Scripts\python.exe ..\thai-doc-ocr\train.py ..\thai-doc-ocr\configs\qwen3vl_2b_qlora_train.yaml
.venv\Scripts\llamafactory-cli.exe export ..\thai-doc-ocr\configs\export_tuned.yaml

# แปลงเป็น GGUF ใส่ Ollama ทั้งโมเดลก่อนเทรนและหลังเทรน ด้วยวิธีเดียวกัน
cd ..\thai-doc-ocr
..\LlamaFactory\.venv\Scripts\python.exe to_ollama.py Qwen/Qwen3-VL-2B-Instruct qwen3vl-base-q8
..\LlamaFactory\.venv\Scripts\python.exe to_ollama.py ..\LlamaFactory\models\qwen3vl-2b-thaidoc qwen3vl-thaidoc-q8

# วัดผล
..\.venv\Scripts\python.exe evaluate.py qwen3vl-thaidoc-q8 --mode whole
..\.venv\Scripts\python.exe evaluate.py qwen3vl-thaidoc-q8 --mode whole --split test_long
```

ให้เทียบผลผ่าน Ollama เท่านั้น เคยลองวัดผ่าน API ของ LlamaFactory แบบ 4-bit (`--api`, `configs/infer_*.yaml`) แล้วอ่านแย่กว่ามาก CER กลาง 0.43 เทียบกับ 0.04 และวนลูป 32%

## ผลล่าสุด (9/11/2026)

| โมเดล (Ollama) | CER กลาง ชุด 50 ฉบับ | ข้อมูลตรง | ภาษาไทยตรง | CER กลาง ชุดยาว | ข้อมูลตรง ชุดยาว |
|---|---|---|---|---|---|
| qwen3-vl:2b-instruct (Q4_K_M) | 0.041 | 82% | 73% | 0.029 | 84% |
| qwen3vl-base-q8 (ก่อนเทรน) | 0.037 | 84% | 77% | 0.035 | 86% |
| qwen3vl-thaidoc-q8 (หลังเทรน) | 0.009 | 95% | 92% | 0.011 | 94% |
| typhoon (Typhoon OCR 1.5 2B, Q4_K_M) | 0.000 | 99.7% | 99.5% | 0.000 | 99.9% |

## แปลงข้อความเป็น JSON (10/11/2026)

ขั้นต่อจากการอ่านรูป: ให้ Qwen3-1.7B (ตัวอักษรอย่างเดียว) แปลงข้อความเป็น JSON ตามรูปแบบด้านล่าง เทรนได้บนการ์ด 4GB ใช้หน่วยความจำประมาณ 2.2GB

```powershell
# ข้อมูลเทรน: JD ที่จำนวนคุณสมบัติแต่ละหมวดไม่ตายตัวและสลับลำดับ เฉลยเรียง key ตามตัวอักษร
..\.venv\Scripts\python.exe generate.py --split train --varied --resumes 300 --jobs 200
..\.venv\Scripts\python.exe to_llamafactory.py --split train_varied --task json

# เทรน รวม adapter แปลงเข้า Ollama และวัดผล 3 ชุดต่อกันเอง (ประมาณ 50 นาที)
powershell -ExecutionPolicy Bypass -File run_json_v2.ps1

# วัดผลเอง: ข้อความจาก Typhoon (ค่าเริ่มต้น) หรือข้อความต้นฉบับ (--source truth)
..\.venv\Scripts\python.exe evaluate_json.py qwen3-1.7b-json-v2-q8 --split test_long
```

**เรียง key ตามตัวอักษร:** เมื่อให้ schema Ollama เขียน key เรียงตามตัวอักษร (`company` ก่อน `title`) ถ้าเทรนด้วยลำดับตามเอกสาร โมเดลจะเติมค่าเลื่อนไปหนึ่งช่อง ตำแหน่งงานกับชื่อบริษัทสลับกัน `to_llamafactory.py` จึงเรียง key ให้ตรงกัน

| Qwen3-1.7B (ข้อมูลถูก / ไม่แต่งเอง) | ก่อนเทรน | เทรนรอบ 1 | เทรนรอบ 2 |
|---|---|---|---|
| ชุด 50 ฉบับ จากข้อความ Typhoon | 89.6 / 88.6 | 95.6 / 96.6 | 97.6 / 98.4 |
| JD 14 ฉบับ | 87.7 / 82.5 | 80.2 / 84.9 | 96.8 / 96.1 |
| เอกสารยาว 20 ฉบับ | 82.2 / 82.7 | 94.3 / 96.9 | 96.7 / 97.4 |
| ชุด 50 ฉบับ จากข้อความต้นฉบับ | 87.9 / 87.7 | 98.2 / 98.7 | 99.1 / 99.6 |

รอบ 1 เทรนด้วยลำดับ key ตามเอกสาร จึงวัดด้วย `--format json` รอบ 2 วัดด้วย schema เหมือนตอนใช้งานจริง ข้อมูลเทรนและชุดทดสอบมาจากตัวสร้างเดียวกัน ผลกับเอกสารจริงน่าจะต่ำกว่านี้

## ไฟล์

- `generate.py` สร้างเอกสารเป็นรูปและ PDF พร้อมเฉลย
- `pools.py` ชื่อ บริษัท มหาวิทยาลัย และตำแหน่งงานที่ใช้สุ่ม
- `evaluate.py` ให้โมเดลอ่านรูป แล้ววัดผล
- `to_llamafactory.py` แปลงข้อมูลเทรนเป็นรูปแบบของ LlamaFactory (ไม่ยอมแปลงชุดทดสอบ)
- `train.py` และ `answer_only_loss.py` เทรนบนการ์ด 4GB
- `to_ollama.py` แปลงโมเดลเป็น GGUF แล้วใส่ Ollama (`--text` สำหรับโมเดลตัวอักษรอย่างเดียว)
- `evaluate_json.py` วัดการแปลงข้อความเป็น JSON ทีละช่อง
- `run_json_v2.ps1` เทรน รวม adapter แปลง และวัดผลโมเดล JSON ต่อกันในคำสั่งเดียว
- `configs/` ค่าตั้งสำหรับเทรน รวม adapter และรันผ่าน API
- `data/test/answers.json` ข้อมูลของแต่ละฉบับ: ข้อความจริง (`text`) และ JSON เฉลย (`answer`)

ห้ามนำ `data/test` ไปเทรน ไม่อย่างนั้นผลวัดจะดีเกินจริง

## ตัววัด

- **CER:** สัดส่วนตัวอักษรที่อ่านผิด ยิ่งต่ำยิ่งดี ไม่นับช่องว่างและเครื่องหมาย Markdown
- **fields_found:** สัดส่วนข้อมูลในเฉลยที่พบตรงตัวในข้อความที่อ่านได้ ยิ่งสูงยิ่งดี

CER ถูกกระทบจากลำดับการอ่าน เอกสารสองคอลัมน์อาจได้ CER สูงทั้งที่อ่านตัวอักษรถูก จึงควรดูคู่กับ fields_found

## รูปแบบ JSON

เรซูเม่:

```json
{
  "doc_type": "resume",
  "name": "", "position": "", "email": "", "phone": "", "location": "", "summary": "",
  "skills": [""],
  "languages": [""],
  "education": [{"degree": "", "field": "", "school": "", "year": ""}],
  "experience": [{"title": "", "company": "", "start": "", "end": null, "details": [""]}]
}
```

`end` เป็น `null` เมื่อยังทำงานอยู่ ปีเก็บตามที่เขียนในเอกสาร เอกสารไทยเป็น พ.ศ.

Job Description:

```json
{
  "doc_type": "job_description",
  "position": "", "company": "", "location": "", "employment_type": "", "salary": "",
  "requirements": {"skills": [""], "abilities": [""], "personality": [""], "other": [""]},
  "responsibilities": [""],
  "benefits": [""],
  "contact": ""
}
```

รูปแบบนี้ตั้งขึ้นเองชั่วคราว ถ้าทีมมีรูปแบบที่ต้องการ ให้แก้ใน `generate.py`
