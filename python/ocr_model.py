import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path

# ----------------------------
# Character Set
# ----------------------------
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK_IDX = len(CHARSET)
NUM_CLASSES = len(CHARSET) + 1


def decode_sequence(seq):
    res = []
    prev = None
    for idx in seq:
        if idx == BLANK_IDX:
            prev = None
            continue
        if idx != prev and idx < len(CHARSET):
            res.append(CHARSET[idx])
        prev = idx
    return "".join(res)


# ----------------------------
# CRNN Model (matches training)
# ----------------------------
class CRNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((4, 2), (4, 1), (0, 1)),
        )

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)          # (B,512,1,W)
        x = x.squeeze(2)         # (B,512,W)
        x = x.permute(0, 2, 1)   # (B,W,512)
        x, _ = self.rnn(x)       # (B,W,512)
        x = self.fc(x)           # (B,W,C)
        x = x.permute(1, 0, 2)   # (T,B,C)
        return x


# ----------------------------
# Load Model
# ----------------------------
def load_ocr_model(device="cpu"):
    path = Path(__file__).parent / "ocr_resnet_ctc.pth"
    model = CRNN(NUM_CLASSES).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# ----------------------------
# OCR Prediction
# ----------------------------
def recognize_plate_text(model, img):
    # img can be PIL or numpy → convert to numpy BGR
    if hasattr(img, 'convert'):
        img = np.array(img)  # convert PIL → numpy

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256, 64))

    img = img.astype("float32") / 255.0
    img = (img - 0.5) / 0.5
    tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        logits = logits[:, 0, :]
        pred = logits.argmax(1).cpu().numpy()
        text = decode_sequence(pred)

    return text
