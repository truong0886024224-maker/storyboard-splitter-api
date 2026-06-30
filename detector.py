import cv2
import numpy as np


def sort_boxes_reading_order(boxes):
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: (b["y1"], b["x1"]))

    rows = []

    for box in boxes:
        placed = False
        box_center_y = (box["y1"] + box["y2"]) / 2

        for row in rows:
            row_center_y = sum((b["y1"] + b["y2"]) / 2 for b in row) / len(row)
            avg_h = sum(b["h"] for b in row) / len(row)

            if abs(box_center_y - row_center_y) < avg_h * 0.45:
                row.append(box)
                placed = True
                break

        if not placed:
            rows.append([box])

    sorted_boxes = []

    for row in rows:
        row = sorted(row, key=lambda b: b["x1"])
        sorted_boxes.extend(row)

    return sorted_boxes


def merge_close_boxes(boxes, distance=12):
    merged = []

    for box in boxes:
        added = False

        for m in merged:
            close_x = abs(box["x1"] - m["x1"]) < distance and abs(box["x2"] - m["x2"]) < distance
            close_y = abs(box["y1"] - m["y1"]) < distance and abs(box["y2"] - m["y2"]) < distance

            if close_x and close_y:
                m["x1"] = min(m["x1"], box["x1"])
                m["y1"] = min(m["y1"], box["y1"])
                m["x2"] = max(m["x2"], box["x2"])
                m["y2"] = max(m["y2"], box["y2"])
                m["w"] = m["x2"] - m["x1"]
                m["h"] = m["y2"] - m["y1"]
                added = True
                break

        if not added:
            merged.append(box)

    return merged


def detect_panels(img):
    H, W = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Tách vùng không phải nền trắng
    _, binary = cv2.threshold(
    gray,
    235,
    255,
    cv2.THRESH_BINARY_INV
)

    # Làm kín vùng panel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Nối các chi tiết trong cùng panel
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    binary = cv2.dilate(binary, kernel2, iterations=1)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    image_area = W * H
    min_area = image_area * 0.003
    max_area = image_area * 0.95

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        if area < min_area:
            continue

        if area > max_area:
            continue

        if w < W * 0.08 or h < H * 0.06:
            continue

        aspect = w / h

        if aspect < 0.15 or aspect > 3.5:
            continue

        pad = 2

        box = {
            "x1": max(0, int(x - pad)),
            "y1": max(0, int(y - pad)),
            "x2": min(W, int(x + w + pad)),
            "y2": min(H, int(y + h + pad)),
        }

        box["x"] = box["x1"]
        box["y"] = box["y1"]
        box["w"] = box["x2"] - box["x1"]
        box["h"] = box["y2"] - box["y1"]

        boxes.append(box)

    boxes = merge_close_boxes(boxes)
    boxes = sort_boxes_reading_order(boxes)

    return boxes
