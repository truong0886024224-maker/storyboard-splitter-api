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
        "message": "Storyboard Splitter Canvas 9x16 No AI No Crop"
    }


def make_grid_boxes(img, rows, cols, margin=2):
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
                "y2": min(h, y2)
            })

    return boxes


def make_background(panel, width, height, bg="black"):
    if bg == "white":
        return np.ones((height, width, 3), dtype=np.uint8) * 255

    if bg == "edge":
        h, w = panel.shape[:2]

        top = panel[:max(2, h // 12), :, :]
        bottom = panel[max(0, h - h // 12):, :, :]
        left = panel[:, :max(2, w // 12), :]
        right = panel[:, max(0, w - w // 12):, :]

        samples = np.concatenate([
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3)
        ], axis=0)

        color = np.mean(samples, axis=0).astype(np.uint8)

        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = color
        return canvas

    return np.zeros((height, width, 3), dtype=np.uint8)


def fit_panel_to_9x16_canvas(panel, width=1080, height=1920, bg="black"):
    h, w = panel.shape[:2]

    scale = min(width / w, height / h)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        panel,
        (new_w, new_h),
        interpolation=cv2.INTER_LANCZOS4
    )

    canvas = make_background(panel, width, height, bg)

    x = (width - new_w) // 2
    y = (height - new_h) // 2

    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas, {
        "mode": "canvas_9x16_no_ai_no_crop",
        "background": bg,
        "original_width": int(w),
        "original_height": int(h),
        "placed_width": int(new_w),
        "placed_height": int(new_h),
        "x": int(x),
        "y": int(y)
    }


def encode_base64(img, quality=96):
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
    rows: int = Query(4),
    cols: int = Query(2),
    margin: int = Query(2),
    width: int = Query(1080),
    height: int = Query(1920),
    quality: int = Query(96),
    bg: str = Query("black")
):
    try:
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

        boxes = make_grid_boxes(img, rows, cols, margin)
        batch_id = str(uuid.uuid4())[:8]
        scenes = []

        for i, box in enumerate(boxes, start=1):
            panel = img[
                box["y1"]:box["y2"],
                box["x1"]:box["x2"]
            ]

            if panel.size == 0:
                continue

            final_img, resize_info = fit_panel_to_9x16_canvas(
                panel,
                width=width,
                height=height,
                bg=bg
            )

            filename = f"{batch_id}_scene_{i:03}_9x16_canvas.jpg"
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
                "resize": resize_info
            })

        return {
            "total": len(scenes),
            "rows": int(rows),
            "cols": int(cols),
            "width": int(width),
            "height": int(height),
            "ratio": "9:16",
            "mode": "canvas_9x16_no_ai_no_crop",
            "ai": "disabled",
            "background": bg,
            "scenes": scenes
        }

    except Exception as e:
        return JSONResponse(
            {"error": "Internal Server Error", "detail": str(e)},
            status_code=500
        )
