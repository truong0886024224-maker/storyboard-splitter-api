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
        "message": "Storyboard Splitter - Keep Full Original Panel 9x16 No AI"
    }


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
                "y2": min(h, y2)
            })

    return boxes


def create_background(img, width, height, bg_mode="edge"):
    h, w = img.shape[:2]

    if bg_mode == "white":
        return np.ones((height, width, 3), dtype=np.uint8) * 255

    if bg_mode == "black":
        return np.zeros((height, width, 3), dtype=np.uint8)

    # bg=edge: lấy màu trung bình từ viền ảnh, không blur, không AI
    top = img[0:max(2, h // 12), :, :]
    bottom = img[max(0, h - h // 12):h, :, :]
    left = img[:, 0:max(2, w // 12), :]
    right = img[:, max(0, w - w // 12):w, :]

    samples = np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3),
    ], axis=0)

    color = np.mean(samples, axis=0).astype(np.uint8)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = color

    return canvas


def make_9x16_keep_full_image(img, width=1080, height=1920, bg_mode="edge"):
    h, w = img.shape[:2]

    # Fit toàn bộ ảnh vào khung 9:16, KHÔNG crop
    scale = min(width / w, height / h)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_LANCZOS4
    )

    canvas = create_background(img, width, height, bg_mode)

    x = (width - new_w) // 2
    y = (height - new_h) // 2

    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas, {
        "mode": "keep_full_original_panel_no_crop",
        "background": bg_mode,
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
    rows: int = Query(4),
    cols: int = Query(2),
    width: int = Query(1080),
    height: int = Query(1920),
    margin: int = Query(0),
    quality: int = Query(96),
    bg: str = Query("edge")
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

        boxes = make_grid_boxes(img, rows, cols, margin)

        batch_id = str(uuid.uuid4())[:8]
        scenes = []

        for i, box in enumerate(boxes, start=1):
            x1 = box["x1"]
            y1 = box["y1"]
            x2 = box["x2"]
            y2 = box["y2"]

            panel = img[y1:y2, x1:x2]

            if panel.size == 0:
                continue

            final_img, debug = make_9x16_keep_full_image(
                panel,
                width=width,
                height=height,
                bg_mode=bg
            )

            filename = f"{batch_id}_scene_{i:03}_9x16_keep_full.jpg"
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
                    "rows": int(rows),
                    "cols": int(cols)
                },
                "storyboard_box": box,
                "resize": debug
            })

        return {
            "total": len(scenes),
            "rows": int(rows),
            "cols": int(cols),
            "width": int(width),
            "height": int(height),
            "ratio": "9:16",
            "ai": "disabled",
            "mode": "keep_full_original_panel_no_crop",
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
