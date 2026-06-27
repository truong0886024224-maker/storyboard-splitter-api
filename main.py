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
        "message": "Storyboard Splitter API Auto Layout Pro is running"
    }


def fallback_grid(img, rows, cols):
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


def border_score(img, rows, cols):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    scores = []

    for c in range(1, cols):
        x = int(w * c / cols)
        band = gray[:, max(0, x - 3):min(w, x + 3)]
        scores.append(float(np.mean(band > 225)))

    for r in range(1, rows):
        y = int(h * r / rows)
        band = gray[max(0, y - 3):min(h, y + 3), :]
        scores.append(float(np.mean(band > 225)))

    return float(np.mean(scores)) if scores else 0.0


def choose_auto_layout(img):
    h, w = img.shape[:2]
    ratio = w / h

    candidates = [
        (4, 2),   # 8 ảnh dọc
        (3, 2),   # 6 ảnh
        (3, 3),   # 9 ảnh
        (4, 3),   # 12 ảnh
        (5, 2),   # 10 ảnh
        (2, 2),   # 4 ảnh
        (2, 3),   # 6 ảnh ngang
        (5, 3),   # 15 ảnh
    ]

    best = None
    best_score = -999

    for rows, cols in candidates:
        score = border_score(img, rows, cols)

        expected_ratio = cols / rows
        ratio_penalty = abs(ratio - expected_ratio) * 0.25

        # Ưu tiên layout phổ biến dọc
        common_bonus = 0
        if (rows, cols) in [(4, 2), (4, 3), (3, 3), (3, 2)]:
            common_bonus = 0.08

        final_score = score - ratio_penalty + common_bonus

        if final_score > best_score:
            best_score = final_score
            best = (rows, cols)

    return best


def detect_layout(img, rows=0, cols=0):
    if rows > 0 and cols > 0:
        return rows, cols

    return choose_auto_layout(img)


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

    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])

    return {
        "cx": int(x + fw // 2),
        "cy": int(y + fh // 2),
        "box": {
            "x": int(x),
            "y": int(y),
            "w": int(fw),
            "h": int(fh)
        }
    }


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
        return int(w // 2), int(h // 2)

    x, y, cw, ch, _ = max(valid, key=lambda b: b[4])

    return int(x + cw // 2), int(y + ch // 2)


def smart_subject_center(img):
    h, w = img.shape[:2]

    face = detect_face_center(img)

    if face:
        cx = face["cx"]
        cy = int(face["cy"] + h * 0.15)
        return cx, cy, "face", face["box"]

    cx, cy = detect_subject_center_by_edges(img)
    return int(cx), int(cy), "edge", None


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

    cropped = img[y1:y2, x1:x2]

    return cropped, {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2)
    }


def resize_and_sharpen(img, width=1080, height=1920):
    cx, cy, method, face_box = smart_subject_center(img)

    cropped, crop_box = crop_9x16_around_center(img, cx, cy)

    if cropped.size == 0:
        cropped = img

    resized = cv2.resize(
        cropped,
        (width, height),
        interpolation=cv2.INTER_LANCZOS4
    )

    blur = cv2.GaussianBlur(resized, (0, 0), 1.0)
    sharp = cv2.addWeighted(resized, 1.4, blur, -0.4, 0)

    debug = {
        "center_x": int(cx),
        "center_y": int(cy),
        "method": method,
        "face_box": face_box,
        "crop_9x16_box": crop_box
    }

    return sharp, debug


@app.post("/split-storyboard")
async def split_storyboard(
    file: UploadFile = File(...),
    rows: int = Query(0),
    cols: int = Query(0),
    width: int = Query(1080),
    height: int = Query(1920)
):
    try:
        contents = await file.read()

        arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(
                {"error": "Cannot read image"},
                status_code=400
            )

        final_rows, final_cols = detect_layout(img, rows, cols)
        boxes = fallback_grid(img, final_rows, final_cols)

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
                "scene": int(i),
                "fileName": filename,
                "mimeType": "image/jpeg",
                "width": int(width),
                "height": int(height),
                "ratio": "9:16",
                "url": f"{BASE_URL}/files/{filename}",
                "layout": {
                    "rows": int(final_rows),
                    "cols": int(final_cols)
                },
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
            "layout": {
                "rows": int(final_rows),
                "cols": int(final_cols)
            },
            "width": int(width),
            "height": int(height),
            "ratio": "9:16",
            "scenes": scenes
        }

    except Exception as e:
        return JSONResponse(
            {
                "error": "Internal Server Error",
                "detail": str(e)
            },
            status_code=500
        )
