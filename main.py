from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import os
import uuid
import base64

app = FastAPI()

OUTPUT_DIR = "files"
BASE_URL = "https://storyboard-splitter-api.onrender.com"

os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="files")


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Storyboard Splitter API - Expanded Crop for AI Outpaint"
    }


def make_grid_boxes(img, rows, cols):
    h, w = img.shape[:2]
    boxes = []

    cell_w = w / cols
    cell_h = h / rows

    for r in range(rows):
        for c in range(cols):
            x1 = int(round(c * cell_w))
            y1 = int(round(r * cell_h))
            x2 = int(round((c + 1) * cell_w))
            y2 = int(round((r + 1) * cell_h))

            # bỏ viền trắng rất nhẹ
            margin = 3

            boxes.append({
                "x1": max(0, x1 + margin),
                "y1": max(0, y1 + margin),
                "x2": min(w, x2 - margin),
                "y2": min(h, y2 - margin)
            })

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

    return float(np.mean(scores)) if scores else 0.0


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


def expand_box(box, img_w, img_h, padding_percent):
    x1 = box["x1"]
    y1 = box["y1"]
    x2 = box["x2"]
    y2 = box["y2"]

    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * padding_percent)
    pad_y = int(bh * padding_percent)

    return {
        "x1": max(0, x1 - pad_x),
        "y1": max(0, y1 - pad_y),
        "x2": min(img_w, x2 + pad_x),
        "y2": min(img_h, y2 + pad_y)
    }


def encode_jpg_base64(img, quality):
    ok, buffer = cv2.imencode(
        ".jpg",
        img,
        [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    )

    if not ok:
        return None

    return base64.b64encode(buffer).decode("utf-8")


@app.post("/split-storyboard")
async def split_storyboard(
    file: UploadFile = File(...),
    rows: int = Query(0),
    cols: int = Query(0),
    padding: float = Query(0.20),
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

        img_h, img_w = img.shape[:2]

        final_rows, final_cols = detect_layout(img, rows, cols)
        original_boxes = make_grid_boxes(img, final_rows, final_cols)

        batch_id = str(uuid.uuid4())[:8]
        scenes = []

        for i, storyboard_box in enumerate(original_boxes, start=1):
            crop_box = expand_box(
                storyboard_box,
                img_w,
                img_h,
                padding
            )

            x1 = crop_box["x1"]
            y1 = crop_box["y1"]
            x2 = crop_box["x2"]
            y2 = crop_box["y2"]

            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            filename = f"{batch_id}_scene_{i:03}_expanded.jpg"
            path = os.path.join(OUTPUT_DIR, filename)

            cv2.imwrite(
                path,
                crop,
                [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
            )

            image_base64 = encode_jpg_base64(crop, quality)

            h, w = crop.shape[:2]

            scenes.append({
                "scene": int(i),
                "fileName": filename,
                "mimeType": "image/jpeg",
                "width": int(w),
                "height": int(h),
                "url": f"{BASE_URL}/files/{filename}",
                "base64": image_base64,
                "layout": {
                    "rows": int(final_rows),
                    "cols": int(final_cols)
                },
                "storyboard_box": storyboard_box,
                "crop_box": crop_box,
                "padding": float(padding)
            })

        return {
            "total": len(scenes),
            "layout": {
                "rows": int(final_rows),
                "cols": int(final_cols)
            },
            "padding": float(padding),
            "quality": int(quality),
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
