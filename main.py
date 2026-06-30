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
        "message": "Storyboard Splitter Detect White Grid Lines"
    }


def pick(form_value, query_value, default):
    return form_value if form_value is not None else query_value if query_value is not None else default


def encode_base64(img, quality):
    ok, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return base64.b64encode(buffer).decode("utf-8") if ok else None


def find_white_lines(gray, axis, threshold=245, min_thickness=3):
    if axis == "x":
        score = np.mean(gray, axis=0)
    else:
        score = np.mean(gray, axis=1)

    white = score >= threshold
    runs = []
    start = None

    for i, is_white in enumerate(white):
        if is_white and start is None:
            start = i
        elif not is_white and start is not None:
            if i - start >= min_thickness:
                runs.append((start, i))
            start = None

    if start is not None and len(white) - start >= min_thickness:
        runs.append((start, len(white)))

    return runs


def pick_separator_near_expected(runs, expected, tolerance):
    candidates = []

    for start, end in runs:
        center = (start + end) // 2
        if abs(center - expected) <= tolerance:
            candidates.append((abs(center - expected), start, end, center))

    if not candidates:
        return expected

    _, start, end, center = min(candidates)

    return center


def detect_edges_by_white_lines(img, rows, cols, line_threshold=245):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    vertical_runs = find_white_lines(
        gray,
        axis="x",
        threshold=line_threshold,
        min_thickness=3
    )

    horizontal_runs = find_white_lines(
        gray,
        axis="y",
        threshold=line_threshold,
        min_thickness=3
    )

    x_edges = [0]
    for c in range(1, cols):
        expected = int(round(w * c / cols))
        tolerance = int(w / cols * 0.35)
        x_edges.append(
            pick_separator_near_expected(vertical_runs, expected, tolerance)
        )
    x_edges.append(w)

    y_edges = [0]
    for r in range(1, rows):
        expected = int(round(h * r / rows))
        tolerance = int(h / rows * 0.35)
        y_edges.append(
            pick_separator_near_expected(horizontal_runs, expected, tolerance)
        )
    y_edges.append(h)

    x_edges = sorted([int(x) for x in x_edges])
    y_edges = sorted([int(y) for y in y_edges])

    return x_edges, y_edges, {
        "vertical_runs": [(int(a), int(b)) for a, b in vertical_runs],
        "horizontal_runs": [(int(a), int(b)) for a, b in horizontal_runs],
        "x_edges": x_edges,
        "y_edges": y_edges
    }


def make_boxes_from_edges(img, rows, cols, margin=0, line_threshold=245):
    h, w = img.shape[:2]

    x_edges, y_edges, debug = detect_edges_by_white_lines(
        img,
        rows,
        cols,
        line_threshold=line_threshold
    )

    boxes = []

    for r in range(rows):
        for c in range(cols):
            x1 = x_edges[c] + margin
            x2 = x_edges[c + 1] - margin
            y1 = y_edges[r] + margin
            y2 = y_edges[r + 1] - margin

            boxes.append({
                "x1": max(0, int(x1)),
                "y1": max(0, int(y1)),
                "x2": min(w, int(x2)),
                "y2": min(h, int(y2))
            })

    return boxes, debug


def trim_white_border(img, threshold=245, pad=0):
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

    x1 = max(0, x + pad)
    y1 = max(0, y + pad)
    x2 = min(img.shape[1], x + w - pad)
    y2 = min(img.shape[0], y + h - pad)

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


def make_canvas_9x16(panel, width=1080, height=1920, bg="black"):
    h, w = panel.shape[:2]

    scale = min(width / w, height / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        panel,
        (new_w, new_h),
        interpolation=cv2.INTER_LANCZOS4
    )

    if bg == "white":
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    elif bg == "edge":
        avg = np.mean(panel.reshape(-1, 3), axis=0).astype(np.uint8)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = avg
    else:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

    x = (width - new_w) // 2
    y = (height - new_h) // 2

    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas, {
        "mode": "canvas_9x16_no_crop",
        "original_width": int(w),
        "original_height": int(h),
        "placed_width": int(new_w),
        "placed_height": int(new_h),
        "x": int(x),
        "y": int(y)
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
    line_threshold_q: Optional[int] = Query(None, alias="line_threshold"),
    trim_panel_q: Optional[bool] = Query(None, alias="trim_panel"),
    trim_threshold_q: Optional[int] = Query(None, alias="trim_threshold"),
    trim_pad_q: Optional[int] = Query(None, alias="trim_pad"),

    rows_f: Optional[int] = Form(None, alias="rows"),
    cols_f: Optional[int] = Form(None, alias="cols"),
    width_f: Optional[int] = Form(None, alias="width"),
    height_f: Optional[int] = Form(None, alias="height"),
    margin_f: Optional[int] = Form(None, alias="margin"),
    quality_f: Optional[int] = Form(None, alias="quality"),
    bg_f: Optional[str] = Form(None, alias="bg"),
    line_threshold_f: Optional[int] = Form(None, alias="line_threshold"),
    trim_panel_f: Optional[bool] = Form(None, alias="trim_panel"),
    trim_threshold_f: Optional[int] = Form(None, alias="trim_threshold"),
    trim_pad_f: Optional[int] = Form(None, alias="trim_pad"),
):
    try:
        rows = int(pick(rows_f, rows_q, 4))
        cols = int(pick(cols_f, cols_q, 3))
        width = int(pick(width_f, width_q, 1080))
        height = int(pick(height_f, height_q, 1920))
        margin = int(pick(margin_f, margin_q, 0))
        quality = int(pick(quality_f, quality_q, 96))
        bg = str(pick(bg_f, bg_q, "black"))

        line_threshold = int(pick(line_threshold_f, line_threshold_q, 245))
        trim_panel = bool(pick(trim_panel_f, trim_panel_q, True))
        trim_threshold = int(pick(trim_threshold_f, trim_threshold_q, 245))
        trim_pad = int(pick(trim_pad_f, trim_pad_q, 0))

        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse({"error": "Cannot read image"}, status_code=400)

        boxes, grid_debug = make_boxes_from_edges(
            img,
            rows,
            cols,
            margin=margin,
            line_threshold=line_threshold
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

            if trim_panel:
                panel, trim_box = trim_white_border(
                    panel,
                    threshold=trim_threshold,
                    pad=trim_pad
                )
            else:
                trim_box = {
                    "x1": 0,
                    "y1": 0,
                    "x2": int(panel.shape[1]),
                    "y2": int(panel.shape[0])
                }

            final_img, resize_info = make_canvas_9x16(
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
            "rows": rows,
            "cols": cols,
            "width": width,
            "height": height,
            "ratio": "9:16",
            "mode": "detect_white_grid_lines_canvas_9x16",
            "ai": "disabled",
            "background": bg,
            "line_threshold": line_threshold,
            "trim_panel": trim_panel,
            "trim_threshold": trim_threshold,
            "trim_pad": trim_pad,
            "grid_debug": grid_debug,
            "scenes": scenes
        }

    except Exception as e:
        return JSONResponse(
            {"error": "Internal Server Error", "detail": str(e)},
            status_code=500
        )
