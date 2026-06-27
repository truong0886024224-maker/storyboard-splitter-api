from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import os
import uuid

app = FastAPI()

OUTPUT_DIR = "files"
BASE_URL = "https://storyboard-splitter-api.onrender.com"

os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="files")


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Storyboard Splitter API Smart Crop is running"
    }


def group_indices(indices, max_gap=5):
    if len(indices) == 0:
        return []

    groups = []
    start = indices[0]
    prev = indices[0]

    for idx in indices[1:]:
        if idx - prev > max_gap:
            groups.append((start, prev))
            start = idx
        prev = idx

    groups.append((start, prev))
    return groups


def find_white_separators(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    vertical_score = np.mean(gray > 235, axis=0)
    horizontal_score = np.mean(gray > 235, axis=1)

    v_candidates = np.where(vertical_score > 0.65)[0]
    h_candidates = np.where(horizontal_score > 0.65)[0]

    v_groups = group_indices(v_candidates, max_gap=6)
    h_groups = group_indices(h_candidates, max_gap=6)

    v_lines = []
    h_lines = []

    for a, b in v_groups:
        thickness = b - a + 1
        if 2 <= thickness <= w * 0.06:
            v_lines.append((a + b) // 2)

    for a, b in h_groups:
        thickness = b - a + 1
        if 2 <= thickness <= h * 0.06:
            h_lines.append((a + b) // 2)

    return v_lines, h_lines


def boxes_from_lines(img, v_lines, h_lines):
    h, w = img.shape[:2]

    xs = [0] + sorted(v_lines) + [w]
    ys = [0] + sorted(h_lines) + [h]

    boxes = []

    for r in range(len(ys) - 1):
        for c in range(len(xs) - 1):
            x1, x2 = xs[c], xs[c + 1]
            y1, y2 = ys[r], ys[r + 1]

            bw = x2 - x1
            bh = y2 - y1

            if bw < w * 0.12 or bh < h * 0.12:
                continue

            margin = 3
            boxes.append((
                max(0, x1 + margin),
                max(0, y1 + margin),
                min(w, x2 - margin),
                min(h, y2 - margin)
            ))

    return boxes


def fallback_grid(img, rows=4, cols=3):
    h, w = img.shape[:2]
    boxes = []

    cell_w = w // cols
    cell_h = h // rows

    for r in range(rows):
        for c in range(cols):
            x1 = c * cell_w
            y1 = r * cell_h
            x2 = (c + 1) * cell_w if c < cols - 1 else w
            y2 = (r + 1) * cell_h if r < rows - 1 else h

            margin = 4
            boxes.append((
                max(0, x1 + margin),
                max(0, y1 + margin),
                min(w, x2 - margin),
                min(h, y2 - margin)
            ))

    return boxes


def detect_storyboard_boxes(img, rows=0, cols=0):
    if rows > 0 and cols > 0:
        return fallback_grid(img, rows, cols)

    v_lines, h_lines = find_white_separators(img)
    boxes = boxes_from_lines(img, v_lines, h_lines)

    if len(boxes) < 4:
        h, w = img.shape[:2]
        ratio = w / h

        if ratio < 0.8:
            boxes = fallback_grid(img, 4, 3)
        else:
            boxes = fallback_grid(img, 4, 2)

    return sorted(boxes, key=lambda b: (b[1], b[0]))


def detect_face_center(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_path)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return None

    # lấy mặt lớn nhất
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    cx = x + w // 2
    cy = y + h // 2

    return cx, cy, (x, y, w, h)


def detect_subject_center_by_edges(img):
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    kernel = np.ones((7, 7), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return w // 2, h // 2

    valid = []

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch

        if area < (w * h) * 0.03:
            continue

        if cw < w * 0.1 or ch < h * 0.1:
            continue

        valid.append((x, y, cw, ch, area))

    if not valid:
        return w // 2, h // 2

    x, y, cw, ch, _ = max(valid, key=lambda b: b[4])

    cx = x + cw // 2
    cy = y + ch // 2

    return cx, cy


def smart_subject_center(img):
    h, w = img.shape[:2]

    face = detect_face_center(img)

    if face:
        cx, cy, box = face

        # Nếu có mặt thì ưu tiên mặt, nhưng crop sẽ đặt mặt hơi cao hơn trung tâm
        smart_cy = int(cy + h * 0.15)
        return cx, smart_cy, "face", box

    cx, cy = detect_subject_center_by_edges(img)

    return cx, cy, "edge", None


def crop_9x16_around_center(img, center_x, center_y):
    h, w = img.shape[:2]
    target_ratio = 9 / 16

    crop_w = w
    crop_h = int(crop_w / target_ratio)

    if crop_h > h:
        crop_h = h
        crop_w = int(crop_h * target_ratio)

    x1 = int(center_x - crop_w / 2)
    y1 = int(center_y - crop_h / 2)

    x1 = max(0, min(x1, w - crop_w))
    y1 = max(0, min(y1, h - crop_h))

    x2 = x1 + crop_w
    y2 = y1 + crop_h

    return img[y1:y2, x1:x2], {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2)
    }


def resize_and_sharpen(img, width=1080, height=1920):
    cx, cy, method, face_box = smart_subject_center(img)

    cropped, crop_box = crop_9x16_around_center(img, cx, cy)

    resized = cv2.resize(
        cropped,
        (width, height),
        interpolation=cv2.INTER_LANCZOS4
    )

    # sharpen nhẹ, tránh làm ảnh giả
    blur = cv2.GaussianBlur(resized, (0, 0), 1.0)
    sharp = cv2.addWeighted(resized, 1.45, blur, -0.45, 0)

    return sharp, {
        "center_x": int(cx),
        "center_y": int(cy),
        "method": method,
        "face_box": face_box,
        "crop_9x16_box": crop_box
    }


@app.post("/split-storyboard")
async def split_storyboard(
    file: UploadFile = File(...),
    rows: int = Query(0),
    cols: int = Query(0),
    width: int = Query(1080),
    height: int = Query(1920)
):
    contents = await file.read()

    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(
            {"error": "Cannot read image"},
            status_code=400
        )

    boxes = detect_storyboard_boxes(img, rows, cols)

    batch_id = str(uuid.uuid4())[:8]
    scenes = []

    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        frame = img[y1:y2, x1:x2]

        if frame.size == 0:
            continue

        final_img, debug = resize_and_sharpen(frame, width, height)

        filename = f"{batch_id}_scene_{i:03}_9x16_{width}x{height}.jpg"
        path = os.path.join(OUTPUT_DIR, filename)

        cv2.imwrite(path, final_img, [cv2.IMWRITE_JPEG_QUALITY, 96])

        scenes.append({
            "scene": i,
            "fileName": filename,
            "mimeType": "image/jpeg",
            "width": width,
            "height": height,
            "ratio": "9:16",
            "url": f"{BASE_URL}/files/{filename}",
            "storyboard_box": {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2)
            },
            "smart_crop": debug
        })

    return {
        "total": len(scenes),
        "width": width,
        "height": height,
        "ratio": "9:16",
        "scenes": scenes
    }
