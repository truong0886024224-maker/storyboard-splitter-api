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
    return {"status": "ok", "message": "Storyboard Splitter Exact 9x16 No AI"}


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


def crop_to_9x16(img):
    h, w = img.shape[:2]
    target_ratio = 9 / 16

    current_ratio = w / h

    if current_ratio > target_ratio:
        # ảnh đang quá ngang, cắt hai bên
        new_w = int(h * target_ratio)
        x1 = (w - new_w) // 2
        x2 = x1 + new_w
        y1 = 0
        y2 = h
    else:
        # ảnh đang quá cao, cắt trên dưới
        new_h = int(w / target_ratio)
        y1 = (h - new_h) // 2
        y2 = y1 + new_h
        x1 = 0
        x2 = w

    return img[y1:y2, x1:x2], {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2)
    }


def resize_sharp(img, width=1080, height=1920):
    resized = cv2.resize(
        img,
        (width, height),
        interpolation=cv2.INTER_LANCZOS4
    )

    blur = cv2.GaussianBlur(resized, (0, 0), 0.8)
    sharp = cv2.addWeighted(resized, 1.2, blur, -0.2, 0)

    return sharp


@app.post("/split-storyboard")
async def split_storyboard(
    file: UploadFile = File(...),
    rows: int = Query(4),
    cols: int = Query(2),
    width: int = Query(1080),
    height: int = Query(1920),
    quality: int = Query(96),
    margin: int = Query(2)
):
    try:
        contents = await file.read()
        arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse({"error": "Cannot read image"}, status_code=400)

        boxes = make_grid_boxes(img, rows, cols, margin)

        batch_id = str(uuid.uuid4())[:8]
        scenes = []

        for i, box in enumerate(boxes, start=1):
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

            panel = img[y1:y2, x1:x2]

            if panel.size == 0:
                continue

            crop_9x16, crop_box = crop_to_9x16(panel)
            final_img = resize_sharp(crop_9x16, width, height)

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
                    "rows": int(rows),
                    "cols": int(cols)
                },
                "storyboard_box": box,
                "crop_9x16_box_inside_panel": crop_box
            })

        return {
            "total": len(scenes),
            "rows": int(rows),
            "cols": int(cols),
            "width": int(width),
            "height": int(height),
            "ratio": "9:16",
            "ai": "disabled",
            "mode": "exact_crop_from_storyboard",
            "scenes": scenes
        }

    except Exception as e:
        return JSONResponse(
            {"error": "Internal Server Error", "detail": str(e)},
            status_code=500
        )
