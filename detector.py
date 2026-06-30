import cv2
import numpy as np


def sort_boxes(boxes):
    boxes = sorted(boxes, key=lambda b: (b["y"], b["x"]))

    rows = []
    for box in boxes:
        placed = False
        for row in rows:
            avg_y = sum(b["y"] for b in row) / len(row)
            if abs(box["y"] - avg_y) < box["h"] * 0.5:
                row.append(box)
                placed = True
                break

        if not placed:
            rows.append([box])

    sorted_boxes = []
    for row in rows:
        row = sorted(row, key=lambda b: b["x"])
        sorted_boxes.extend(row)

    return sorted_boxes


def detect_panels(img):
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Tìm vùng không trắng
    mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    min_area = (w * h) * 0.01

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh

        if area < min_area:
            continue

        if bw < w * 0.12 or bh < h * 0.08:
            continue

        boxes.append({
            "x": int(x),
            "y": int(y),
            "w": int(bw),
            "h": int(bh),
            "x1": int(x),
            "y1": int(y),
            "x2": int(x + bw),
            "y2": int(y + bh)
        })

    boxes = sort_boxes(boxes)

    return boxes
