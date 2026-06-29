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
        "message": "Storyboard Splitter 9x16 No AI"
    }


def make_grid_boxes(img, rows, cols):
    h, w = img.shape[:2]
    boxes = []

    cell_w = w / cols
    cell_h = h / rows

    for r in range(rows):
        for c in range(cols):
            x1 = int(round(c * cell_w)) + 4
            y1 = int(round(r * cell_h)) + 4
            x2 = int(round((c + 1) * cell_w)) - 4
            y2 = int(round((r + 1) * cell_h)) - 4

            boxes.append((
                max(0, x1),
                max(0, y1),
                min(w, x2),
                min(h, y2)
            ))

    return boxes


def border_score(img, rows, cols):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scores = []

    for c in range(1, cols):
        x = int(w * c / cols)
        band = gray[:, max(0, x - 5):min(w, x + 5)]
        scores.append(float(np.mean(band > 225)))

    for r in range(1, rows):
        y = int(h * r / rows)
        band = gray[max(0, y - 5):min(h, y + 5), :]
        scores.append(float(np.mean(band > 225)))

    return float(np.mean(scores)) if scores else 0


def choose_auto_layout(img):
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

    best = (4, 2)
    best_score = -999

    for rows, cols in candidates:
        score = border_score(img, rows, cols)

        if score > best_score:
            best_score = score
            best = (rows, cols)

    return best


def detect_layout(img, rows, cols):
    if rows > 0 and cols > 0:
        return rows, cols

    return choose_auto_layout(img)


def sharpen(img):
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    return cv2.addWeighted(img, 1.25, blur, -0.25, 0)


def get_edge_background_color(img):
    h, w = img.shape[:2]

    top = img[0:max(2, h // 10), :, :]
    bottom = img[max(0, h - h // 10):h, :, :]
    left = img[:, 0:max(2, w // 10), :]
    right = img[:, max(0, w - w // 10):w, :]

    samples = np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3),
    ], axis=0)

    return np.mean(samples, axis=0).astype(np.uint8)


def convert_to_9x16_no_ai(img, width=1080, height=1920, bg="edge"):
    h, w = img.shape[:2]

    scale = min(width / w, height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_LANCZOS4
    )

    resized = sharpen(resized)

    if bg == "white":
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    elif bg == "black":
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        color = get_edge_background_color(img)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = color

    x = (width - new_w) // 2
    y = (height - new_h) // 2

    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas, {
        "mode": "no_ai_keep_original",
        "background": bg,
        "original_width": int(w),
        "original_height": int(h),
        "final_width": int(width),
        "final_height": int(height),
        "placed_width": int(new_w),
        "placed_height": int(new_h),
        "x": int(x),
        "y": int(y)
    }


@app.post("/split-storyboard")
async def split_storyboard(
    file: UploadFile = File(...),
    rows: int = Query(0),
    cols: int = Query(0),
    width: int = Query(1080),
    height: int = Query(1920),
    bg: str = Query("edge"),
    quality: int = Query(96)
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
        boxes = make_grid_boxes(img, final_rows, final_cols)

        batch_id = str(uuid.uuid4())[:8]
        scenes = []

        for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
            frame = img[y1:y2, x1:x2]

            if frame.size == 0:
                continue

            final_img, debug = convert_to_9x16_no_ai(
                frame,
                width=width,
                height=height,
                bg=bg
            )

            filename = f"{batch_id}_scene_{i:03}_9x16.jpg"
            path = os.path.join(OUTPUT_DIR, filename)

            cv2.imwrite(
                path,
                final_img,
                [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
            )

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
            "ai": "disabled",
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
