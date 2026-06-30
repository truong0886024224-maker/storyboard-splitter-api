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
        "message": "Storyboard Splitter V3 rows-cols trim canvas"
    }


def pick(form_value, query_value, default):
    if form_value is not None:
        return form_value
    if query_value is not None:
        return query_value
    return default


def encode_base64(img, quality=96):
    ok, buffer = cv2.imencode(
        ".jpg",
        img,
        [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    )
    if not ok:
        return None
    return base64.b64encode(buffer).decode("utf-8")


def make_grid_boxes(img, rows, cols, margin=0):
    h, w = img.shape[:2]
    boxes = []

    cell_w = w / cols
    cell_h = h / rows

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


def trim_border(img, threshold=245, trim_pad=0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = gray < threshold
    coords = cv2.findNonZero(mask.astype(np.uint8))

    if coords is None:
        return img, {
            "x1": 0,
            "y1": 0,
            "x2": int(img.shape[1]),
            "y2": int(img.shape[0])
        }

    x, y, w, h = cv2.boundingRect(coords)

    x1 = max(0, x + trim_pad)
    y1 = max(0, y + trim_pad)
    x2 = min(img.shape[1], x + w - trim_pad)
    y2 = min(img.shape[0], y + h - trim_pad)

    if x2 <= x1 or y2 <= y1:
        return img, {
            "x1": 0,
            "y1": 0,
            "x2": int(img.shape[1]),
            "y2": int(img.shape[0])
        }

    return img[y1:y2, x1:x2], {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2)
    }


def make_background(img, width, height, bg="black"):
    if bg == "white":
        return np.ones((height, width, 3), dtype=np.uint8) * 255

    if bg == "edge":
        avg = np.mean(img.reshape(-1, 3), axis=0).astype(np.uint8)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = avg
        return canvas

    return np.zeros((height, width, 3), dtype=np.uint8)


def fit_to_canvas(img, width=1080, height=1920, bg="black"):
    h, w = img.shape[:2]

    scale = min(width / w, height / h)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_LANCZOS4
    )

    canvas = make_background(img, width, height, bg)

    x = (width - new_w) // 2
    y = (height - new_h) // 2

    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas, {
        "mode": "contain_no_crop_no_ai",
        "original_width": int(w),
        "original_height": int(h),
        "placed_width": int(new_w),
        "placed_height": int(new_h),
        "x": int(x),
        "y": int(y)
    }


def fit_cover(img, width=1080, height=1920):
    h, w = img.shape[:2]

    scale = max(width / w, height / h)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_LANCZOS4
    )

    x = max(0, (new_w - width) // 2)
    y = max(0, (new_h - height) // 2)

    cropped = resized[y:y + height, x:x + width]

    return cropped, {
        "mode": "cover_crop",
        "original_width": int(w),
        "original_height": int(h),
        "resized_width": int(new_w),
        "resized_height": int(new_h),
        "crop_x": int(x),
        "crop_y": int(y)
    }


@app.post("/split-storyboard")
async def split_storyboard(
    file: UploadFile = File(...),

    rows_q: Optional[int] = Query(None, alias="rows"),
    cols_q: Optional[int] = Query(None, alias="cols"),
    width_q: Optional[int] = Query(None, alias="width"),
    height_q: Optional[int] = Query(None, alias="height"),
    margin_q: Optional[int] = Query(None, alias="margin"),
    quality_q: Optional[int] = Query(None, alias="quality"),
    bg_q: Optional[str] = Query(None, alias="bg"),
    mode_q: Optional[str] = Query(None, alias="mode"),
    trim_q: Optional[bool] = Query(None, alias="trim"),
    trim_threshold_q: Optional[int] = Query(None, alias="trim_threshold"),
    trim_pad_q: Optional[int] = Query(None, alias="trim_pad"),

    rows_f: Optional[int] = Form(None, alias="rows"),
    cols_f: Optional[int] = Form(None, alias="cols"),
    width_f: Optional[int] = Form(None, alias="width"),
    height_f: Optional[int] = Form(None, alias="height"),
    target_width_f: Optional[int] = Form(None, alias="target_width"),
    target_height_f: Optional[int] = Form(None, alias="target_height"),
    margin_f: Optional[int] = Form(None, alias="margin"),
    quality_f: Optional[int] = Form(None, alias="quality"),
    bg_f: Optional[str] = Form(None, alias="bg"),
    mode_f: Optional[str] = Form(None, alias="mode"),
    trim_f: Optional[bool] = Form(None, alias="trim"),
    trim_threshold_f: Optional[int] = Form(None, alias="trim_threshold"),
    trim_pad_f: Optional[int] = Form(None, alias="trim_pad"),
):
    try:
        rows = int(pick(rows_f, rows_q, 4))
        cols = int(pick(cols_f, cols_q, 2))

        width = int(target_width_f or width_f or width_q or 1080)
        height = int(target_height_f or height_f or height_q or 1920)

        margin = int(pick(margin_f, margin_q, 0))
        quality = int(pick(quality_f, quality_q, 96))
        bg = str(pick(bg_f, bg_q, "black"))
        mode = str(pick(mode_f, mode_q, "contain"))

        trim = bool(pick(trim_f, trim_q, True))
        trim_threshold = int(pick(trim_threshold_f, trim_threshold_q, 245))
        trim_pad = int(pick(trim_pad_f, trim_pad_q, 0))

        contents = await file.read()
        img = cv2.imdecode(
            np.frombuffer(contents, np.uint8),
            cv2.IMREAD_COLOR
        )

        if img is None:
            return JSONResponse(
                {"error": "Cannot read image"},
                status_code=400
            )

        boxes = make_grid_boxes(
            img,
            rows=rows,
            cols=cols,
            margin=margin
        )

        batch_id = str(uuid.uuid4())[:8]
        scenes = []

        for i, box in enumerate(boxes, start=1):
            panel = img[
                box["y1"]:box["y2"],
                box["x1"]:box["x2"]
            ]

            if panel.size == 0:
                continue

            if trim:
                panel, trim_box = trim_border(
                    panel,
                    threshold=trim_threshold,
                    trim_pad=trim_pad
                )
            else:
                trim_box = {
                    "x1": 0,
                    "y1": 0,
                    "x2": int(panel.shape[1]),
                    "y2": int(panel.shape[0])
                }

            if mode == "cover":
                final_img, resize_info = fit_cover(
                    panel,
                    width=width,
                    height=height
                )
            else:
                final_img, resize_info = fit_to_canvas(
                    panel,
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
                "base64": encode_base64(final_img, quality),
                "storyboard_box": box,
                "panel_trim_box": trim_box,
                "resize": resize_info
            })

        return {
            "total": len(scenes),
            "rows": int(rows),
            "cols": int(cols),
            "width": int(width),
            "height": int(height),
            "ratio": "9:16",
            "mode": mode,
            "ai": "disabled",
            "background": bg,
            "trim": trim,
            "trim_threshold": trim_threshold,
            "trim_pad": trim_pad,
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
