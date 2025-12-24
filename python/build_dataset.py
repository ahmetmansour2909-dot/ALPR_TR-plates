from pathlib import Path
import cv2
from ultralytics import YOLO
import torch

# --- patch لـ torch.load عشان يفتح وزن YOLO بدون مشكلة weights_only ---
_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat
# -----------------------------------------------------------------------

# مسارات المجلدات (عدّلها إذا اختلفت عندك)
RAW_CARS_DIR = Path(r"C:\ALPR_project\ALPR_datasets\raw cars")
TRAIN_IMAGES_DIR = Path(r"C:\ALPR_project\ALPR_datasets\train\images")
TRAIN_LABEL_FILE = Path(r"C:\ALPR_project\ALPR_datasets\train\train_label.txt")

MODEL_PATH = Path(r"C:\ALPR_project\python\license_plate_detector.pt")

# تأكد أن المجلدات موجودة
TRAIN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_LABEL_FILE.parent.mkdir(parents=True, exist_ok=True)

# تحميل نموذج YOLO
print(f"Loading YOLO model from: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))

# نبدأ الترقيم من عدد الصور الموجودة أصلاً (لو كان هناك صور قديمة)
existing = sorted(TRAIN_IMAGES_DIR.glob("plate_*.jpg"))
start_index = 1
if existing:
    # آخر رقم موجود + 1
    last = existing[-1].stem.replace("plate_", "")
    try:
        start_index = int(last) + 1
    except ValueError:
        pass

print(f"Starting index: {start_index}")

# نفتح ملف الـ labels في وضع الإضافة
label_f = TRAIN_LABEL_FILE.open("a", encoding="utf-8")

supported_ext = {".jpg", ".jpeg", ".png", ".bmp"}

idx = start_index

for img_path in sorted(RAW_CARS_DIR.iterdir()):
    if img_path.suffix.lower() not in supported_ext:
        continue

    print(f"\nProcessing: {img_path.name}")

    image = cv2.imread(str(img_path))
    if image is None:
        print("  -> Failed to read image, skipping.")
        continue

    # تشغيل YOLO
    results = model(image)
    if not results or len(results[0].boxes) == 0:
        print("  -> No plate detected, skipping.")
        continue

    # نأخذ أعلى ثقة
    best_box = max(results[0].boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = best_box.xyxy[0].tolist()
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    plate_img = image[y1:y2, x1:x2]
    if plate_img.size == 0:
        print("  -> Empty crop, skipping.")
        continue

    out_name = f"plate_{idx:04d}.jpg"
    out_path = TRAIN_IMAGES_DIR / out_name

    # حفظ صورة اللوحة
    cv2.imwrite(str(out_path), plate_img)
    print(f"  -> Saved cropped plate as: {out_name}")

    # نضيف سطر في ملف الـ labels مع placeholder
    # لاحقاً ستفتح الملف وتضع مكان LABEL_HERE نص اللوحة الصحيح
    label_f.write(f"train/images/{out_name} LABEL_HERE\n")

    idx += 1

label_f.close()

print("\nDone!")
print("الآن افتح الملف التالي وعدّل LABEL_HERE لكل سطر إلى نص اللوحة الصحيح:")
print(TRAIN_LABEL_FILE)
