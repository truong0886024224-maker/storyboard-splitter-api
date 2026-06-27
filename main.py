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
        "message": "Storyboard Splitter No AI - Keep Original"
    }


def make_grid_boxes(img, rows, cols):
    h, w = img.shape[:2]
    boxes = []

    cell_w = w / cols
    cell_h = h / rows

    for r in range(rows):
        for c in range(cols):
            x1 = int(round(c * cell_w)) + 3
            y1 = int(round(r * cell_h)) + 3
            x2 = int(round((c + 1) * cell_w)) - 3
            y2 = int(round((r + 1) * cell_h)) - 3

            boxes.append({
                "x1": max(0, x1),
                "y1": max(0, y1),
                "x2": min(w, x2),
                "y2": min(h, y2)
            })

    return boxes


def make_9x16_no_ai(img, width=1080, height=1920, bg="black"):
    h, w = img.shape[:2]

    scale = min(width / w, height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_LANCZOS4
    )

    if bg == "white":
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    elif bg == "average":
        avg_color = np.mean(img.reshape(-1, 3), axis=0).astype(np.uint8)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = avg_color
    else:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

    x = (width - new_w) // 2
    y = (height - new_h) // 2

    canvas[y:y + new_h, x:x + new_w] = resized

    # sharpen nhẹ, không vẽ thêm
    blur = cv2.GaussianBlur(canvas, (0, 0), 1.0)
    sharp = cv2.addWeighted(canvas, 1.25, blur, -0.25, 0)

    return sharp


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
    width: int = Query(1080),
    height: int = Query(1920),
    bg: str = Query("black"),
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

        boxes = make_grid_boxes(img, rows, cols)
        batch_id = str(uuid.uuid4())[:8]

        scenes = []

        for i, box in enumerate(boxes, start=1):
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            final_img = make_9x16_no_ai(
                crop,
                width=width,
                height=height,
                bg=bg
            )

            filename = f"{batch_id}_scene_{i:03}_9x16_no_ai.jpg"
            path = os.path.join(OUTPUT_DIR, filename)

            cv2.imwrite(
                path,
                final_img,
                [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
            )

            image_base64 = encode_base64(final_img, quality)

            scenes.append({
                "scene": int(i),
                "fileName": filename,
                "mimeType": "image/jpeg",
                "width": int(width),
                "height": int(height),
                "ratio": "9:16",
                "url": f"{BASE_URL}/files/{filename}",
                "base64": image_base64,
                "storyboard_box": box
            })

        return {
            "total": len(scenes),
            "rows": int(rows),
            "cols": int(cols),
            "width": int(width),
            "height": int(height),
            "ratio": "9:16",
            "background": bg,
            "ai": "disabled",
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
