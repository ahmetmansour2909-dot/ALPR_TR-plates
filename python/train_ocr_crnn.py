import os
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ======================
# إعداد الحروف (نفس اللي باللوحات)
# ======================
# نسمح بكل الأرقام + كل الحروف الإنجليزية الكبيرة
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# blank token لـ CTC يكون index 0
char2idx = {c: i + 1 for i, c in enumerate(CHARS)}
idx2char = {i + 1: c for i, c in enumerate(CHARS)}
blank_idx = 0
num_classes = len(CHARS) + 1  # + blank

# ======================
# Dataset
# ======================

class PlateOCRDataset(Dataset):
    def __init__(self, labels_file, root_dir):
        self.samples = []
        self.root_dir = Path(root_dir)

        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path, label = line.split(" ")
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def encode_label(self, text):
        indices = []
        for ch in text:
            if ch in char2idx:
                indices.append(char2idx[ch])
            else:
                # تجاهل أي حرف غير موجود في القائمة
                print("Warning: unknown char in label:", ch)
        return torch.tensor(indices, dtype=torch.long)

    def __getitem__(self, idx):
        rel_path, label_text = self.samples[idx]
        img_path = self.root_dir / rel_path

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Cannot read image: {img_path}")

        # الحجم الموحد للصور: 32x160
        img = cv2.resize(img, (160, 32))
        img = img.astype(np.float32) / 255.0
        img = img[None, :, :]  # (1, H, W)

        label = self.encode_label(label_text)

        return torch.from_numpy(img), label


def collate_fn(batch):
    # batch: list of (image_tensor, label_tensor)
    images = [b[0] for b in batch]
    labels = [b[1] for b in batch]

    images = torch.stack(images, dim=0)  # (B, 1, H, W)

    # ندمج جميع الlabels في Tensor واحد
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_cat = torch.cat(labels, dim=0)

    return images, labels_cat, label_lengths

# ======================
# نموذج CRNN بسيط
# ======================

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # CNN لتصغير الصورة واستخراج الميزات
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),  # (B,64,32,160)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),              # (B,64,16,80)

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),              # (B,128,8,40)

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),    # (B,256,4,40)

            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 1), (2, 1)),    # (B,512,2,40)

            nn.Conv2d(512, 512, 2, padding=0),  # (B,512,1,39) تقريبًا
            nn.ReLU(True)
        )

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (B, 1, H, W)
        x = self.cnn(x)  # (B, 512, 1, W_new)
        b, c, h, w = x.size()
        assert h == 1
        x = x.squeeze(2)  # (B, 512, W_new)
        x = x.permute(2, 0, 1)  # (T, B, 512) where T = W_new

        x, _ = self.rnn(x)  # (T, B, 512)
        x = self.fc(x)      # (T, B, num_classes)
        return x

# ======================
# دوال مساعدة لفك التنبؤ (للتقييم)
# ======================

def ctc_decode(logits):  # logits: (T,B,C)
    # نأخذ argmax على C
    preds = logits.softmax(2).argmax(2)  # (T,B)
    preds = preds.transpose(0, 1)        # (B,T)
    texts = []
    for seq in preds:
        prev = blank_idx
        s = []
        for p in seq.cpu().numpy().tolist():
            if p != prev and p != blank_idx:
                if p in idx2char:
                    s.append(idx2char[p])
            prev = p
        texts.append("".join(s))
    return texts

# ======================
# التدريب
# ======================

def train_ocr():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    base = Path(__file__).resolve().parent
    dataset_root = base.parent / "ALPR_datasets"

    train_labels = dataset_root / "train" / "train_label.txt"
    val_labels = dataset_root / "val" / "val_label.txt"

    train_ds = PlateOCRDataset(str(train_labels), dataset_root)
    val_ds = PlateOCRDataset(str(val_labels), dataset_root)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False,
                            collate_fn=collate_fn)

    model = CRNN(num_classes=num_classes).to(device)
    criterion = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 40

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels, label_lengths in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)

            optimizer.zero_grad()
            logits = model(images)  # (T,B,C)
            T, B, C = logits.size()

            log_probs = logits.log_softmax(2)

            input_lengths = torch.full(size=(B,), fill_value=T,
                                       dtype=torch.long, device=device)

            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))

        # تقييم بسيط على val
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels, label_lengths in val_loader:
                images = images.to(device)
                logits = model(images)
                preds_text = ctc_decode(logits)

                # إعادة فك labels الحقيقية للمقارنة
                gt_texts = []
                offset = 0
                for L in label_lengths:
                    seq = labels[offset:offset + L]
                    offset += L
                    text = "".join(idx2char[i.item()] for i in seq)
                    gt_texts.append(text)

                for p, g in zip(preds_text, gt_texts):
                    total += 1
                    if p == g:
                        correct += 1

        val_acc = correct / max(1, total)
        print(f"Epoch {epoch}/{num_epochs} - loss: {avg_loss:.4f} - val_acc: {val_acc:.3f}")

    # حفظ النموذج
    out_path = base / "ocr_crnn.pth"
    torch.save(model.state_dict(), out_path)
    print("Model saved to:", out_path)


if __name__ == "__main__":
    train_ocr()
