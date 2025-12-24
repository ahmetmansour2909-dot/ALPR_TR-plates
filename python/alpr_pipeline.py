# alpr_pipeline.py
# -----------------
# YOLO + OCR pipeline لقراءة لوحة من صورة سيارة واحدة

import sys
from pathlib import Path
import json

import cv2
import numpy as np
import torch

# ================== PATCH FOR PyTorch 2.6 ==================
_ORIG_TORCH_LOAD = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _ORIG_TORCH_LOAD(*args, **kwargs)
torch.load = _patched_torch_load
# ===========================================================

from ultralytics import YOLO
from ocr_model import load_ocr_model, recognize_plate_text


# ---- إعدادات عامة ----
YOLO_WEIGHTS = "license_plate_detector.pt"
OUTPUT_DIR = Path("outputs")


# -----------------------------------------------------------
# 1) قراءة مسار الصورة من سطر الأوامر
# -----------------------------------------------------------
def load_image_path_from_argv() -> str:
    if len(sys.argv) < 2:
        print("[ERROR] Please pass an image path.")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.is_file():
        print(f"[ERROR] Image does not exist: {img_path}")
        sys.exit(1)

    print(f"[INFO] Using image: {img_path}")
    return str(img_path)


# -----------------------------------------------------------
# 2) تحميل الموديلات (YOLO + OCR)
# -----------------------------------------------------------
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    print(f"[INFO] Loading YOLO weights from: {YOLO_WEIGHTS}")
    yolo = YOLO(YOLO_WEIGHTS)

    print("[INFO] Loading OCR model...")
    ocr_model = load_ocr_model(device)

    return yolo, ocr_model, device


# -----------------------------------------------------------
# 3) اختيار أفضل بوكس للوحة
# -----------------------------------------------------------
def select_best_plate_box(results):
    if results.boxes is None or len(results.boxes) == 0:
        return None

    confs = results.boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))

    box_xyxy = results.boxes.xyxy[best_idx].cpu().numpy().astype(int)
    return box_xyxy


# -----------------------------------------------------------
# 4) تشغيل الـ pipeline على صورة واحدة
# -----------------------------------------------------------
def run_pipeline_on_image(img_path, yolo_model, ocr_model, device):
    OUTPUT_DIR.mkdir(exist_ok=True)

    # قراءة الصورة الأصلية
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    # ----- YOLO detection -----
    print("[INFO] Running YOLO inference...")
    yolo_results = yolo_model(img_path)[0]

    best_box = select_best_plate_box(yolo_results)
    if best_box is None:
        print("[INFO] No plate detected.")
        return {
            "plate_text": "",
            "ocr_confidence": 0.0,
            "car_image": "",
            "plate_image": "",
            "message": "no_plate_detected",
        }

    x1, y1, x2, y2 = best_box.tolist()

    # تأكد من صحة الإحداثيات
    h, w = bgr.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))

    plate_crop = bgr[y1:y2, x1:x2].copy()

    # ----- OCR -----
    if plate_crop.size == 0:
        print("[WARN] Empty crop. Skipping OCR.")
        plate_text = ""
        ocr_conf = 0.0
    else:
        print("[INFO] Running OCR on cropped plate...")
        plate_text = recognize_plate_text(ocr_model, plate_crop)
        ocr_conf = 0.99  # ثابت مؤقت

    # ----- حفظ الصور -----
    img_stem = Path(img_path).stem

    annotated = yolo_results.plot()
    car_out_path = OUTPUT_DIR / f"{img_stem}_yolo.jpg"
    cv2.imwrite(str(car_out_path), annotated)

    plate_out_path = OUTPUT_DIR / f"{img_stem}_plate.jpg"
    cv2.imwrite(str(plate_out_path), plate_crop)

    print(f"[INFO] Saved car image: {car_out_path}")
    print(f"[INFO] Saved cropped plate: {plate_out_path}")
    print(f"[INFO] OCR TEXT = {plate_text}")

    return {
        "plate_text": plate_text,
        "ocr_confidence": float(ocr_conf),
        "car_image": str(car_out_path).replace("\\", "/"),
        "plate_image": str(plate_out_path).replace("\\", "/"),
        "message": "ok",
    }


# -----------------------------------------------------------
# 5) main
# -----------------------------------------------------------
def main():
    img_path = load_image_path_from_argv()
    yolo_model, ocr_model, device = load_models()

    result = run_pipeline_on_image(img_path, yolo_model, ocr_model, device)

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
