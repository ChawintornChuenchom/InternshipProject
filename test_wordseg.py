import sys, platform, time
from openthai_latte_wordseg_lite import tokenize, tokenize_batch, tokenize_nbest, Segmenter

print("OS:", platform.platform(), "| Python:", sys.version.split()[0])

cases = [
    "ตากลม","นั่งตากลม","เด็กตากลมนั่งตากลม", "ไปหาหมอนวด", "ข้อความของคุณ",
    "ใช้ Python 3.11 พัฒนา OCR",
    "ทำงานที่บริษัท ABC จำกัด ตำแหน่ง Data Engineer",
    "ติดต่อ test@example.com หรือ https://example.com/a?b=1",
    "เบอร์โทร 081-234-5678 ราคา 1,500.50 บาท",
    "สวัสดีครับ","สะง่อง","แลคตาซอยห้าบาท","คริสเตียโน่ โรนัลโด้",
    "บันดาลลงบันได้บันทึกไว้จำจงดี","จราจรจรจรวล",
    "สุนธรภู่ครูกวีขี่ฮอนด้า","อีลอนมัสขับเทสล่าโก้นักหนา",
    "เจนเซ่นหวงควงอีกสามตามกันมา","มาร์กเมต้าหุ้นไม่ขึ้นกลุ้มใจจริง",
    "สนุกมากๆ ดีๆ", "ไม่ไหวแล้วนะะะ", "สวัสดี 😊 ครับ",
    "กรุงเทพมหานคร", "มหาวิทยาลัยสงขลานครินทร์",
    "ก็ก้ก๊ก๋", "ฯพณฯ ๑๒๓ ๆ ฯลฯ",
    "line1\nline2 ไทย\r\nบรรทัดสาม\tแท็บ",
    "  นำหน้าและตามหลัง  ", "ไทย  สองช่องว่าง   สามช่อง",
    "English only 123", "日本語 中文 ไทย", "", "   ",
    "I ว่า I จะกลับไป America ช่วงประมาณ December 2023 ",
    "สวัสดีครับ 😊 ผมชื่อ John Doe อายุ 30 ปี",
    "ติดต่อผมได้ที่ example@example.com",
    "ห้า😊H😊A😊5😊๕",
    "วันนี้ฉันไปกินข้าวที่โรงอาหาร"

]

print("\n== tokenize 3 ระดับ + round-trip ==")
for text in cases:
    for unit in (None, "long", "short", "minimal"):
        try:
            t = tokenize(text) if unit is None else tokenize(text, unit=unit)
            ok = "".join(t) == text
            print(f"{text[:30]!r:35} {str(unit):8} ok={ok} {t}")
        except Exception as e:
            print(f"{text[:30]!r:35} {unit} ERROR {type(e).__name__}: {e}")

print("\n== nbest ==")
for text in ["ตากลม", "ไปหาหมอนวด", "", "abc"]:
    for n in (1, 3, 0, -1, 100):
        try:
            print(repr(text), "n =", n, tokenize_nbest(text, n=n))
        except Exception as e:
            print(repr(text), "n =", n, "ERROR", type(e).__name__, e)

print("\n== batch เทียบกับเรียกทีละบรรทัด ==")
try:
    batch = tokenize_batch(cases)
    single = [tokenize(c) for c in cases]
    print("batch == single:", batch == single)
    for c, b, s in zip(cases, batch, single):
        if b != s:
            print("  ต่างกัน:", repr(c), b, s)
    print("batch []:", tokenize_batch([]))
except Exception as e:
    print("batch ERROR", type(e).__name__, e)

print("\n== Segmenter user_words ==")
try:
    seg = Segmenter(user_words=["จ๊อดแฟร์", "ก", "ก" * 20])
    print(seg.tokenize("ไปงานจ๊อดแฟร์มา"))
    print(seg.tokenize("ไปงานจ๊อดแฟร์มา", unit="long"))
    print(tokenize("ไปงานจ๊อดแฟร์มา"))
except Exception as e:
    print("Segmenter ERROR", type(e).__name__, e)

print("\n== ข้อความยาว ==")
for size in (1000, 10000, 10001, 200000):
    text = "ฉันไปโรงเรียน Python 3.11 " * (size // 20)
    try:
        t0 = time.time()
        t = tokenize(text)
        print(size, len(text), "ok=", "".join(t) == text, f"{time.time()-t0:.2f}s")
    except Exception as e:
        print(size, "ERROR", type(e).__name__, e)

print("\n== input ผิดชนิด ==")
for bad in (None, 123, ["ตากลม"], b"abc"):
    try:
        print(repr(bad), tokenize(bad))
    except Exception as e:
        print(repr(bad), "ERROR", type(e).__name__, e)

print("\n== unit ผิด ==")
try:
    print(tokenize("ตากลม", unit="xxx"))
except Exception as e:
    print("ERROR", type(e).__name__, e)