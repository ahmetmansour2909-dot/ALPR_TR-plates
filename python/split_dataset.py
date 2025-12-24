import os
import random
import shutil

# المسارات الأساسية
DATASET_DIR = r"C:\ALPR_project\ALPR_datasets"
IMAGES_DIR = os.path.join(DATASET_DIR, "train", "images")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

TRAIN_LABEL = os.path.join(TRAIN_DIR, "train_label.txt")
VAL_LABEL = os.path.join(VAL_DIR, "val_label.txt")
TEST_LABEL = os.path.join(TEST_DIR, "test_label.txt")

# قراءة كل أسماء الصور من train/images
all_images = [f for f in os.listdir(IMAGES_DIR)
              if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

if not all_images:
    raise RuntimeError("No images found in train/images")

random.shuffle(all_images)
total = len(all_images)

train_count = int(total * 0.8)
val_count = int(total * 0.1)
test_count = total - train_count - val_count

train_imgs = all_images[:train_count]
val_imgs = all_images[train_count:train_count + val_count]
test_imgs = all_images[train_count + val_count:]

print(f"Total images: {total}")
print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

# نقرأ الـ labels الأصلية (اللي كتبتها يدويًا لكل صورة)
original_label_file = TRAIN_LABEL  # هو نفسه train_label.txt الموجود الآن
if not os.path.exists(original_label_file):
    raise RuntimeError(f"Label file not found: {original_label_file}")

label_dict = {}
with open(original_label_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        path, label = line.split(" ")
        fname = os.path.basename(path)
        label_dict[fname] = label

# دالة تكتب ملف label لقائمة صور معيّنة
def write_labels(img_list, outfile, folder_name):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w", encoding="utf-8") as f:
        for img in img_list:
            if img in label_dict:
                f.write(f"{folder_name}/images/{img} {label_dict[img]}\n")

# نكتب الملفات الثلاثة
write_labels(train_imgs, TRAIN_LABEL, "train")
write_labels(val_imgs, VAL_LABEL, "val")
write_labels(test_imgs, TEST_LABEL, "test")

# ننسخ صور val و test فقط إلى مجلداتهم
def copy_images(img_list, dest_folder):
    dest_images = os.path.join(dest_folder, "images")
    os.makedirs(dest_images, exist_ok=True)
    for img in img_list:
        src = os.path.join(IMAGES_DIR, img)
        dst = os.path.join(dest_images, img)
        if not os.path.exists(dst):  # لو منسوخة من قبل لا نعيدها
            shutil.copy(src, dst)

copy_images(val_imgs, VAL_DIR)
copy_images(test_imgs, TEST_DIR)

print("Dataset split completed successfully.")
print("Files created:")
print(f"  {TRAIN_LABEL}")
print(f"  {VAL_LABEL}")
print(f"  {TEST_LABEL}")
