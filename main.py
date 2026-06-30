from fastapi import FastAPI, UploadFile, File, Query, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import os
import uuid
import base64
from typing import Optional

app = FastAPI()

OUTPUT_DIR = "files"
BASE_URL = "https://storyboard-splitter-api.onrender.com"

os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="files")


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Storyboard Splitter Exact 9x16 High Quality"
    }


def pick(form_value, query_value, default):
    return form_value if form_value is not None else query_value if query_value is not None else default


def encode_base64(img, quality):
    ok, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return base64.b64encode(buffer).decode("utf-8") if ok else None


def make_grid_boxes(img, rows, cols, margin=0):
    h, w = img.shape[:2]
    cell_w = w / cols
    cell_h = h / rows
    boxes = []

    for r in range(rows):
        for c in range(cols):
            x1 = int(round(c * cell_w)) + margin
            y1 = int(round(r * cell_h)) + margin
            x2 = int(round((c + 1) * cell_w)) - margin
            y2 = int(round((r + 1) * cell_h)) - margin

            boxes.append({
                "x1": max(0, x1),
                "y1": max(0, y1),
                "x2": min(w, x2),
                "y2": min(h, y2),
            })

    return boxes


def trim_white_border(img, threshold=245, pad=0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray < threshold
    coords = cv2.findNonZero(mask.astype(np.uint8))

    if coords is None:
        return img

    x, y, w, h = cv2.boundingRect(coords)

    x1 = max(0, x + pad)
    y1 = max(0, y + pad)
    x2 = min(img.shape[1], x + w - pad)
    y2 = min(img.shape[0], y + h - pad)

    if x2 <= x1 or y2 <= y1:
        return img

    return img[y1:y2, x1:x2]


def fit_exact_9x16(img, width=1080, height=1920, mode="contain", bg="black"):
    h, w = img.shape[:2]

    if mode == "cover":
        scale = max(width / w, height / h)
    else:
        scale = min(width / w, height / h)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    if mode == "cover":
        x1 = max(0, (new_w - width) // 2)
        y1 = max(0, (new_h - height) // 2)
        return resized[y1:y1 + height, x1:x1 + width]

    if bg == "white":
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    else:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

    x = (width - new_w) // 2
    y = (height - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas


@app.post("/split-storyboard")
async def split_storyboard(
    file: UploadFile = File(...),

    rows_q: Optional[int] = Query(None, alias="rows"),
    cols_q: Optional[int] = Query(None, alias="cols"),
    width_q: Optional[int] = Query(None, alias="width"),
    height_q: Optional[int] = Query(None, alias="height"),
    margin_q: Optional[int] = Query(None, alias="margin"),
    quality_q: Optional[int] = Query(None, alias="quality"),
    trim_q: Optional[bool] = Query(None, alias="trim"),
    trim_threshold_q: Optional[int] = Query(None, alias="trim_threshold"),
    trim_pad_q: Optional[int] = Query(None, alias="trim_pad"),
    mode_q: Optional[str] = Query(None, alias="mode"),
    bg_q: Optional[str] = Query(None, alias="bg"),

    rows_f: Optional[int] = Form(None, alias="rows"),
    cols_f: Optional[int] = Form(None, alias="cols"),
    width_f: Optional[int] = Form(None, alias="width"),
    height_f: Optional[int] = Form(None, alias="height"),
    margin_f: Optional[int] = Form(None, alias="margin"),
    quality_f: Optional[int] = Form(None, alias="quality"),
    trim_f: Optional[bool] = Form(None, alias="trim"),
    trim_threshold_f: Optional[int] = Form(None, alias="trim_threshold"),
    trim_pad_f: Optional[int] = Form(None, alias="trim_pad"),
    mode_f: Optional[str] = Form(None, alias="mode"),
    bg_f: Optional[str] = Form(None, alias="bg"),
):
    try:
        rows = int(pick(rows_f, rows_q, 4))
        cols = int(pick(cols_f, cols_q, 3))
        width = int(pick(width_f, width_q, 1080))
        height = int(pick(height_f, height_q, 1920))
        margin = int(pick(margin_f, margin_q, 0))
        quality = int(pick(quality_f, quality_q, 96))
        trim = bool(pick(trim_f, trim_q, True))
        trim_threshold = int(pick(trim_threshold_f, trim_threshold_q, 245))
        trim_pad = int(pick(trim_pad_f, trim_pad_q, 0))
        mode = str(pick(mode_f, mode_q, "contain"))
        bg = str(pick(bg_f, bg_q, "black"))

        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse({"error": "Cannot read image"}, status_code=400)

        boxes = make_grid_boxes(img, rows, cols, margin)

        batch_id = str(uuid.uuid4())[:8]
        scenes = []

        for i, box in enumerate(boxes, start=1):
            panel = img[box["y1"]:box["y2"], box["x1"]:box["x2"]]

            if panel.size == 0:
                continue

            if trim:
                panel = trim_white_border(panel, trim_threshold, trim_pad)

            final_img = fit_exact_9x16(
                panel,
                width=width,
                height=height,
                mode=mode,
                bg=bg
            )

            filename = f"{batch_id}_scene_{i:03}_9x16.jpg"
            path = os.path.join(OUTPUT_DIR, filename)

            cv2.imwrite(path, final_img, [cv2.IMWRITE_JPEG_QUALITY, quality])

            scenes.append({
                "scene": i,
                "fileName": filename,
                "mimeType": "image/jpeg",
                "width": width,
                "height": height,
                "ratio": "9:16",
                "url": f"{BASE_URL}/files/{filename}",
                "base64": encode_base64(final_img, quality),
                "storyboard_box": box,
                "mode": mode
            })

        return {
            "total": len(scenes),
            "rows": rows,
            "cols": cols,
            "width": width,
            "height": height,
            "ratio": "9:16",
            "mode": mode,
            "ai": "disabled",
            "scenes": scenes
        }

    except Exception as e:
        return JSONResponse({"error": "Internal Server Error", "detail": str(e)}, status_code=500)
