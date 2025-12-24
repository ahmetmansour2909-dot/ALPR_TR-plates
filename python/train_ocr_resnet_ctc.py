import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
#  الأبجدية (أرقام + حروف لاتينية)
# -----------------------------
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK_IDX = len(CHARSET)
NUM_CLASSES = len(CHARSET) + 1  # + blank


def clean_label(text: str) -> str:
    text = text.strip().upper()
    out = []
    for c in text:
        if c in CHARSET:
            out.append(c)
    return "".join(out)


def encode_label(text: str):
    return [CHARSET.index(c) for c in text if c in CHARSET]


def decode_sequence(seq):
    res = []
    prev = None
    for idx in seq:
        if idx == BLANK_IDX:
            prev = None
            continue
        if idx != prev:
            if 0 <= idx < len(CHARSET):
                res.append(CHARSET[idx])
        prev = idx
    return "".join(res)

# -----------------------------
#   Dataset
# -----------------------------


class PlateDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train"):
        """
        root_dir = C:/ALPR_project/ALPR_datasets
        split    = train or val
        """
        self.root = Path(root_dir)
        self.split = split

        label_file = self.root / split / f"{split}_label.txt"
        self.samples = []

        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # مثال: train/images/553_jpg....jpg 34FUZ323
                try:
                    path_part, text = line.split(maxsplit=1)
                except ValueError:
                    continue
                text = clean_label(text)
                if len(text) == 0:
                    continue
                img_path = self.root / path_part.replace("/", os.sep)
                if img_path.is_file():
                    self.samples.append((img_path, text))

        print(f"[{split}] samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: Path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.ones((64, 256), dtype=np.uint8) * 255
        img = cv2.resize(img, (256, 64))
        img = img.astype("float32") / 255.0
        img = (img - 0.5) / 0.5  # [-1,1]
        img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        return img

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        img = self._load_image(img_path)
        label_indices = encode_label(text)
        label = torch.tensor(label_indices, dtype=torch.long)
        return img, label, len(label), text


def collate_fn(batch):
    imgs, labels, label_lengths, raw_texts = zip(*batch)
    imgs = torch.stack(imgs, dim=0)  # (B, 1, 64, 256)
    labels = torch.cat(labels, dim=0)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    return imgs, labels, label_lengths, list(raw_texts)

# -----------------------------
#   CRNN model (CNN + BiLSTM)
# -----------------------------


class CRNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # أبسط من قبل (نخفف التعقيد)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 32x128

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 16x64

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 8x64

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 4x64

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((4, 2), (4, 1), (0, 1)),  # 1x64
        )

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (B,1,64,256)
        x = self.cnn(x)             # (B,512,1,W)
        b, c, h, w = x.size()
        x = x.squeeze(2)            # (B,512,W)
        x = x.permute(0, 2, 1)      # (B,W,512)
        x, _ = self.rnn(x)          # (B,W,512)
        x = self.fc(x)              # (B,W,C)
        x = x.permute(1, 0, 2)      # (T,B,C) للـ CTC
        return x

# -----------------------------
#   Training loop
# -----------------------------


def train_ocr():
    ROOT = r"C:\ALPR_project\ALPR_datasets"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_ds = PlateDataset(ROOT, "train")
    val_ds = PlateDataset(ROOT, "val")

    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    model = CRNN(NUM_CLASSES).to(device)
    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)

    EPOCHS = 40
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for imgs, labels, label_lengths, raw_texts in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)              # (T,B,C)
            log_probs = F.log_softmax(logits, dim=2)

            T, B, C = log_probs.size()
            input_lengths = torch.full((B,), T, dtype=torch.long).to(device)

            loss = criterion(log_probs, labels, input_lengths, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch}/{EPOCHS} - train_loss: {avg_loss:.4f}")

        # ---- تقييم بسيط على val + طباعة بعض الأمثلة ----
        model.eval()
        correct = 0
        total = 0
        examples_shown = 0
        with torch.no_grad():
            for imgs, labels, label_lengths, raw_texts in val_loader:
                imgs = imgs.to(device)
                logits = model(imgs)          # (T,B,C)
                log_probs = F.log_softmax(logits, dim=2)
                preds = log_probs.argmax(2)   # (T,B)

                T, B = preds.size()
                preds = preds.permute(1, 0)   # (B,T)

                for i in range(B):
                    pred_seq = preds[i].cpu().numpy().tolist()
                    pred_text = decode_sequence(pred_seq)
                    true_text = clean_label(raw_texts[i])
                    if len(true_text) > 0:
                        total += 1
                        if pred_text == true_text:
                            correct += 1

                    # نطبع بعض الأمثلة في أول batch من كل epoch
                    if examples_shown < 5:
                        print(f"  [VAL] true='{true_text}'  pred='{pred_text}'")
                        examples_shown += 1

                if examples_shown >= 5:
                    # ما نطبع كل شيء، فقط أول batch
                    break

        acc = correct / total if total > 0 else 0.0
        print(f"           val_acc (exact match): {acc:.3f}")

    out_path = Path(__file__).parent / "ocr_resnet_ctc.pth"
    torch.save(model.state_dict(), out_path)
    print(f"Model saved to: {out_path}")


if __name__ == "__main__":
    train_ocr()
