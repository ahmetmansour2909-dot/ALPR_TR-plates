from pathlib import Path
import shutil
import yaml

# -----------------------------
# CONFIG
# -----------------------------
ROBOFLOW_ROOT = Path(r"C:\ALPR_project\external_datasets")
ALPR_ROOT = Path(r"C:\ALPR_project\ALPR_datasets")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load class mapping
# -----------------------------
data_yaml_path = ROBOFLOW_ROOT / "data.yaml"
with open(data_yaml_path, "r", encoding="utf-8") as f:
    data_cfg = yaml.safe_load(f)

names = data_cfg.get("names")

# Roboflow may output list or dict
if isinstance(names, list):
    id2char = {i: names[i] for i in range(len(names))}
elif isinstance(names, dict):
    id2char = {int(k): v for k, v in names.items()}
else:
    raise ValueError("Unexpected format for names in data.yaml")

print("Loaded classes:", id2char)

# -----------------------------
# Convert function
# -----------------------------
def process_split(split_name):
    images_dir = ROBOFLOW_ROOT / split_name / "images"
    labels_dir = ROBOFLOW_ROOT / split_name / "labels"

    if not images_dir.exists():
        print(f"[WARN] Missing split: {split_name}")
        return

    out_images_dir = ALPR_ROOT / split_name / "images"
    ensure_dir(out_images_dir)

    label_txt_path = ALPR_ROOT / split_name / f"{split_name}_label.txt"
    out_file = open(label_txt_path, "w", encoding="utf-8")

    for label_file in sorted(labels_dir.glob("*.txt")):
        content = label_file.read_text().strip().splitlines()
        chars = []

        for line in content:
            parts = line.split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            x_center = float(parts[1])

            ch = id2char.get(cls_id, "")
            if ch:
                chars.append((x_center, ch))

        if not chars:
            continue

        chars_sorted = sorted(chars, key=lambda x: x[0])
        plate_text = "".join([c[1] for c in chars_sorted])

        stem = label_file.stem  # example: "1_jpg.rf.xxxxx"

        # extract image path
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            test = images_dir / f"{stem}{ext}"
            if test.exists():
                img_path = test
                break

        if img_path is None:
            print("[WARN] Missing image for:", stem)
            continue

        dst = out_images_dir / img_path.name
        shutil.copy2(img_path, dst)

        rel_path = f"{split_name}/images/{img_path.name}"
        out_file.write(f"{rel_path} {plate_text}\n")

    out_file.close()
    print(f"[OK] Finished {split_name}, labels saved to: {label_txt_path}")

# -----------------------------
# RUN FOR TRAIN + VALID
# -----------------------------
process_split("train")
process_split("valid")

print("\n[DONE] Dataset conversion complete.")
