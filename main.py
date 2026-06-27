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
        "message": "Storyboard Splitter API Keep Original is running"
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
        (4, 2),
        (3, 2),
        (3, 3),
        (4, 3),
        (5, 2),
        (2, 2),
        (2, 3),
        (5, 3),
    ]

    best = None
    best_score = -999

    for rows, cols in candidates:
        score = border_score(img, rows, cols)

        expected_ratio = cols / rows
        ratio_penalty = abs(ratio - expected_ratio) * 0.25

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


def resize_keep_original(img, width=1080, height=1920, bg_mode="black"):
    h, w = img.shape[:2]

    scale = min(width / w, height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_LANCZOS4
    )

    if bg_mode == "blur":
        bg = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        canvas = cv2.GaussianBlur(bg, (0, 0), 30)
    else:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

    x = (width - new_w) // 2
    y = (height - new_h) // 2

    canvas[y:y + new_h, x:x + new_w] = resized

    blur = cv2.GaussianBlur(canvas, (0, 0), 1.0)
    sharp = cv2.addWeighted(canvas, 1.25, blur, -0.25, 0)

    debug = {
        "mode": "keep_original_no_crop",
        "background": bg_mode,
        "original_width": int(w),
        "original_height": int(h),
        "new_width": int(new_w),
        "new_height": int(new_h),
        "x": int(x),
        "y": int(y)
    }

    return sharp, debug


@app.post("/split-storyboard")
async def split_storyboard(
    file: UploadFile = File(...),
    rows: int = Query(0),
    cols: int = Query(0),
    width: int = Query(1080),
    height: int = Query(1920),
    bg: str = Query("black")
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

            final_img, debug = resize_keep_original(frame, width, height, bg)

            filename = f"{batch_id}_scene_{i:03}_keep_{width}x{height}.jpg"
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
                "resize": debug
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
            "background": bg,
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
