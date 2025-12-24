import sys
import os
from pathlib import Path

import torch

# ================== PATCH FOR PyTorch 2.6 ==================
# هنا نعدل torch.load بحيث يحمّل الـ weights بالطريقة القديمة
# (weights_only=False) حتى لا تظهر مشكلة Weights only load failed
_orig_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    # إذا ما أرسلت weights_only، نخليها False تلقائيًا
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = patched_torch_load
# ===========================================================

from ultralytics import YOLO
import cv2


# ممكن تغيّرها لاحقًا لو عندك وزن مخصص للوحات مثلاً "best.pt"
YOLO_WEIGHTS = "yolov8n.pt"


def load_image_path_from_argv() -> str:
    """يقرأ مسار الصورة من سطر الأوامر ويتأكد أن الملف موجود."""
    if len(sys.argv) < 2:
        print("[ERROR] Please pass an image path, e.g.:")
        print('       py alpr_yolo.py "..\\ALPR_datasets\\raw cars\\car13.jpg"')
        sys.exit(1)

    arg_path = sys.argv[1]
    print(f"[INFO] Image path argument: {arg_path}")

    img_path = Path(arg_path)

    if not img_path.is_file():
        print(f"[ERROR] Image does not exist: {img_path}")
        sys.exit(1)

    return str(img_path)


def run_yolo_on_image(img_path: str) -> None:
    """يشغّل YOLO على صورة واحدة ويحفظ النتيجة مع البوكسات."""

    print("[INFO] Loading YOLO model...")
    model = YOLO(YOLO_WEIGHTS)  # لو الملف مش موجود هيقوم بتحميله من الإنترنت

    print("[INFO] Running inference...")
    results = model(img_path)[0]  # أول نتيجة (صورة واحدة)

    # طباعة معلومات عن البوكسات
    if results.boxes is None or len(results.boxes) == 0:
        print("[INFO] No objects detected.")
    else:
        print(f"[INFO] Detected {len(results.boxes)} objects.")
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, str(cls_id))
            print(
                f"  #{i}: {cls_name} "
                f"conf={conf:.3f} "
                f"box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})"
            )

    # حفظ الصورة مع البوكسات
    annotated = results.plot()  # numpy BGR image
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / ("yolo_" + Path(img_path).name)
    cv2.imwrite(str(out_path), annotated)
    print(f"[INFO] Saved annotated image to: {out_path}")


def main():
    img_path = load_image_path_from_argv()
    run_yolo_on_image(img_path)


if __name__ == "__main__":
    main()
