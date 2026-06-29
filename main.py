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
        "message": "Storyboard Splitter - detect white grid lines - canvas 9x16"
    }


def pick(form_value, query_value, default):
    return form_value if form_value is not None else query_value if query_value is not None else default


def encode_base64(img, quality):
    ok, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return base64.b64encode(buffer).decode("utf-8") if ok else None


def find_white_runs(score, threshold=245, min_size=4):
    is_white = score >= threshold
    runs = []
    start = None

    for i, v in enumerate(is_white):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_size:
                runs.append((start, i))
            start = None

    if start is not None and len(is_white) - start >= min_size:
        runs.append((start, len(is_white)))

    return runs


def clean_runs(runs, min_gap=20):
    if not runs:
        return []

    merged = [runs[0]]

    for s, e in runs[1:]:
        ps, pe = merged[-1]
        if s - pe <= min_gap:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))

    return merged


def detect_grid_lines(img, rows, cols):
    """
    Detect white separator lines.
    Return x_edges, y_edges.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    col_score = np.mean(gray, axis=0)
    row_score = np.mean(gray, axis=1)

    vertical_runs = clean_runs(find_white_runs(col_score, threshold=242, min_size=3), min_gap=8)
    horizontal_runs = clean_runs(find_white_runs(row_score, threshold=242, min_size=3), min_gap=8)

    # Lấy line separator theo vị trí gần dự kiến
    x_edges = [0]
    for c in range(1, cols):
        expected = int(w * c / cols)
        candidates = []
        for s, e in vertical_runs:
            center = (s + e) // 2
            if abs(center - expected) < w / cols * 0.35:
                candidates.append((abs(center - expected), s, e))
        if candidates:
            _, s, e = min(candidates)
            x_edges.append((s + e) // 2)
        else:
            x_edges.append(expected)
    x_edges.append(w)

    y_edges = [0]
    for r in range(1, rows):
        expected = int(h * r / rows)
        candidates = []
        for s, e in horizontal_runs:
            center = (s + e) // 2
            if abs(center - expected) < h / rows * 0.35:
                candidates.append((abs(center - expected), s, e))
        if candidates:
            _, s, e = min(candidates)
            y_edges.append((s + e) // 2)
        else:
            y_edges.append(expected)
    y_edges.append(h)

    x_edges = sorted(list(set([int(x) for x in x_edges])))
    y_edges = sorted(list(set([int(y) for y in y_edges])))

    return x_edges, y_edges


def trim_white_border(panel, threshold=245, pad=0):
    """
    Remove white frame inside each panel.
    """
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    mask = gray < threshold

    coords = cv2.findNonZero(mask.astype(np.uint8))
    if coords is None:
        return panel, {"x1": 0, "y1": 0, "x2": panel.shape[1], "y2": panel.shape[0]}

    x, y, w, h = cv2.boundingRect(coords)

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(panel.shape[1], x + w + pad)
    y2 = min(panel.shape[0], y + h + pad)

    return panel[y1:y2, x1:x2], {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2)
    }


def make_boxes_by_detected_lines(img, rows, cols, margin=0):
    h, w = img.shape[:2]
    x_edges, y_edges = detect_grid_lines(img, rows, cols)

    # Nếu detect không đủ cạnh thì fallback chia đều
    if len(x_edges) != cols + 1:
        x_edges = [int(round(i * w / cols)) for i in range(cols + 1)]
    if len(y_edges) != rows + 1:
        y_edges = [int(round(i * h / rows)) for i in range(rows + 1)]

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
                "y2": min(h, int(y2)),
            })

    return boxes, {
        "x_edges": [int(x) for x in x_edges],
        "y_edges": [int(y) for y in y_edges]
    }


def make_background(panel, width, height, bg):
    if bg == "white":
        return np.ones((height, width, 3), dtype=np.uint8) * 255
    if bg == "edge":
        avg = np.mean(panel.reshape(-1, 3), axis=0).astype(np.uint8)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = avg
        return canvas
    return np.zeros((height, width, 3), dtype=np.uint8)


def fit_to_9x16_canvas(panel, width=1080, height=1920, bg="black"):
    h, w = panel.shape[:2]

    scale = min(width / w, height / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(panel, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    canvas = make_background(panel, width, height, bg)

    x = (width - new_w) // 2
    y = (height - new_h) // 2

    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas, {
        "mode": "canvas_9x16_no_ai_no_crop",
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
    trim_panel_q: Optional[bool] = Query(None, alias="trim_panel"),

    rows_f: Optional[int] = Form(None, alias="rows"),
    cols_f: Optional[int] = Form(None, alias="cols"),
    width_f: Optional[int] = Form(None, alias="width"),
    height_f: Optional[int] = Form(None, alias="height"),
    target_width_f: Optional[int] = Form(None, alias="target_width"),
    target_height_f: Optional[int] = Form(None, alias="target_height"),
    margin_f: Optional[int] = Form(None, alias="margin"),
    quality_f: Optional[int] = Form(None, alias="quality"),
    bg_f: Optional[str] = Form(None, alias="bg"),
    trim_panel_f: Optional[bool] = Form(None, alias="trim_panel"),
):
    try:
        rows = int(pick(rows_f, rows_q, 5))
        cols = int(pick(cols_f, cols_q, 2))
        width = int(target_width_f or width_f or width_q or 1080)
        height = int(target_height_f or height_f or height_q or 1920)
        margin = int(pick(margin_f, margin_q, 0))
        quality = int(pick(quality_f, quality_q, 96))
        bg = str(pick(bg_f, bg_q, "black"))
        trim_panel = bool(pick(trim_panel_f, trim_panel_q, True))

        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse({"error": "Cannot read image"}, status_code=400)

        boxes, grid_debug = make_boxes_by_detected_lines(img, rows, cols, margin)

        batch_id = str(uuid.uuid4())[:8]
        scenes = []

        for i, box in enumerate(boxes, start=1):
            raw_panel = img[box["y1"]:box["y2"], box["x1"]:box["x2"]]

            if raw_panel.size == 0:
                continue

            if trim_panel:
                panel, trim_box = trim_white_border(raw_panel, threshold=245, pad=0)
            else:
                panel = raw_panel
                trim_box = {
                    "x1": 0,
                    "y1": 0,
                    "x2": int(raw_panel.shape[1]),
                    "y2": int(raw_panel.shape[0])
                }

            final_img, resize_info = fit_to_9x16_canvas(
                panel,
                width=width,
                height=height,
                bg=bg
            )

            filename = f"{batch_id}_scene_{i:03}_9x16.jpg"
            path = os.path.join(OUTPUT_DIR, filename)

            cv2.imwrite(path, final_img, [cv2.IMWRITE_JPEG_QUALITY, quality])

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
            "trim_panel": trim_panel,
            "grid_debug": grid_debug,
            "scenes": scenes
        }

    except Exception as e:
        return JSONResponse(
            {"error": "Internal Server Error", "detail": str(e)},
            status_code=500
        )
