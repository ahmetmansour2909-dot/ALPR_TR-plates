import random
import shutil
from pathlib import Path
from zipfile import ZipFile

# كم صورة نريد أخذها من ال ZIP ؟
NUM_IMAGES = 500  # تقدر تغيّرها 300 / 600 حسب ما تحب

base = Path(__file__).resolve().parent
dataset_root = base.parent / "ALPR_datasets"

# مسار ملف ال ZIP الذي نقلناه قبل قليل
zip_path = base.parent / "external_datasets" / "synthetic_tr.zip"

train_img_dir = dataset_root / "train" / "images"
val_img_dir   = dataset_root / "val" / "images"
test_img_dir  = dataset_root / "test" / "images"

train_label_file = dataset_root / "train" / "train_label.txt"
val_label_file   = dataset_root / "val" / "val_label.txt"
test_label_file  = dataset_root / "test" / "test_label.txt"


def main():
    if not zip_path.exists():
        print("ZIP file not found:", zip_path)
        return

    # تأكد أن مجلدات الصور موجودة
    for d in [train_img_dir, val_img_dir, test_img_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with ZipFile(zip_path, "r") as zf:
        # كل الملفات داخل ال ZIP
        all_names = zf.namelist()
        # نختار فقط الصور (png/jpg)
        exts = (".png", ".jpg", ".jpeg")
        image_names = [n for n in all_names if n.lower().endswith(exts)]

        if not image_names:
            print("No images found inside ZIP.")
            return

        random.shuffle(image_names)
        selected = image_names[:NUM_IMAGES]
        print(f"Total images in zip: {len(image_names)}")
        print(f"Using {len(selected)} images.")

        # نفتح ملفات ال labels للـ append
        f_train = open(train_label_file, "a", encoding="utf-8")
        f_val   = open(val_label_file, "a", encoding="utf-8")
        f_test  = open(test_label_file, "a", encoding="utf-8")

        for i, member in enumerate(selected):
            # اسم الملف داخل ال ZIP (مثلاً license-plates/34DIB639.png)
            stem = Path(member).stem  # → 34DIB639
            plate_text = stem.upper()

            r = random.random()
            if r < 0.8:
                target_dir = train_img_dir
                f_label = f_train
                prefix = "train"
            elif r < 0.9:
                target_dir = val_img_dir
                f_label = f_val
                prefix = "val"
            else:
                target_dir = test_img_dir
                f_label = f_test
                prefix = "test"

            new_name = f"synt_{i:05d}{Path(member).suffix.lower()}"
            dst_path = target_dir / new_name

            # ننسخ الملف من ال ZIP مباشرة إلى مسارنا
            with zf.open(member) as src, open(dst_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

            rel_path = f"{prefix}/images/{new_name}"
            f_label.write(f"{rel_path} {plate_text}\n")

        f_train.close()
        f_val.close()
        f_test.close()

    print("Done. Synthetic images added to ALPR_datasets and labels appended.")


if __name__ == "__main__":
    main()
