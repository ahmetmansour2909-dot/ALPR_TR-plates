# vehicle_agent.py
# ----------------
# Rule-based post-processing to infer vehicle color and type.

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _load_image(image_path: str):
    path = Path(image_path)
    if not path.is_file():
        return None, f"Image not found: {image_path}"
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None, f"Unable to read image: {image_path}"
    return bgr, ""


def _central_crop(bgr: np.ndarray, crop_ratio: float = 0.6) -> np.ndarray:
    h, w = bgr.shape[:2]
    y0 = int(h * (1 - crop_ratio) / 2)
    y1 = int(h * (1 + crop_ratio) / 2)
    x0 = int(w * (1 - crop_ratio) / 2)
    x1 = int(w * (1 + crop_ratio) / 2)
    if y1 <= y0 or x1 <= x0:
        return bgr
    return bgr[y0:y1, x0:x1]


def _detect_color(bgr: np.ndarray):
    sample = _central_crop(bgr, 0.6)
    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)

    mean_s = float(np.mean(hsv[:, :, 1]))
    mean_v = float(np.mean(hsv[:, :, 2]))

    if mean_s < 35:
        if mean_v > 200:
            color = "white"
        elif mean_v > 150:
            color = "silver"
        elif mean_v > 90:
            color = "gray"
        else:
            color = "black"
        reason = (
            f"Low saturation (mean S={mean_s:.0f}) and brightness V={mean_v:.0f} "
            f"suggest {color}."
        )
        return color, reason

    mask = (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 60)
    hues = hsv[:, :, 0][mask]
    if hues.size == 0:
        color = "gray"
        reason = "No strongly saturated pixels; defaulted to gray."
        return color, reason

    hist = np.bincount(hues.flatten(), minlength=180)
    dominant_hue = int(np.argmax(hist))

    if dominant_hue <= 10 or dominant_hue >= 170:
        color = "red"
    elif dominant_hue <= 20:
        color = "orange"
    elif dominant_hue <= 35:
        color = "yellow"
    elif dominant_hue <= 85:
        color = "green"
    elif dominant_hue <= 135:
        color = "blue"
    elif dominant_hue <= 160:
        color = "purple"
    else:
        color = "red"

    if color in {"orange", "yellow"} and mean_v < 90:
        color = "brown"

    reason = f"Dominant hue {dominant_hue} with mean saturation {mean_s:.0f} -> {color}."
    return color, reason


def _detect_vehicle_type(bgr: np.ndarray):
    h, w = bgr.shape[:2]
    if h == 0 or w == 0:
        return "car", "Empty image; defaulted to car."

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours_info = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    if not contours:
        img_ratio = w / float(h)
        if img_ratio < 1.1:
            vehicle_type = "truck"
        elif img_ratio < 1.45:
            vehicle_type = "SUV"
        else:
            vehicle_type = "car"
        reason = f"No clear contour; used image aspect ratio {img_ratio:.2f} -> {vehicle_type}."
        return vehicle_type, reason

    largest = max(contours, key=cv2.contourArea)
    x, y, box_w, box_h = cv2.boundingRect(largest)
    area_ratio = (box_w * box_h) / float(w * h)
    aspect_ratio = box_w / float(box_h) if box_h else 1.0

    if area_ratio < 0.14 and aspect_ratio > 1.2:
        vehicle_type = "motorcycle"
    elif area_ratio < 0.32:
        vehicle_type = "car"
    elif area_ratio < 0.55:
        vehicle_type = "SUV"
    else:
        vehicle_type = "truck"

    reason = (
        f"Bounding box covers {area_ratio:.2f} of the image with aspect ratio "
        f"{aspect_ratio:.2f} -> {vehicle_type}."
    )
    return vehicle_type, reason


def analyze_vehicle(image_path: str) -> dict:
    bgr, error = _load_image(image_path)
    if bgr is None:
        return {
            "vehicle_color": "unknown",
            "vehicle_type": "unknown",
            "explanation": error or "Image unavailable.",
        }

    color, color_reason = _detect_color(bgr)
    vehicle_type, type_reason = _detect_vehicle_type(bgr)
    explanation = f"{color_reason} {type_reason}"

    return {
        "vehicle_color": color,
        "vehicle_type": vehicle_type,
        "explanation": explanation,
    }


def main() -> None:
    import sys

    image_path = sys.argv[1] if len(sys.argv) > 1 else ""
    result = analyze_vehicle(image_path)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
