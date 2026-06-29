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
    return {"status": "ok", "message": "Storyboard Splitter Extend 9x16 No AI"}


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

            boxes.append((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))

    return boxes


def sharpen(img):
    blur = cv2.GaussianBlur(img, (0, 0), 0.8)
    return cv2.addWeighted(img, 1.18, blur, -0.18, 0)


def extend_to_9x16(panel, width=1080, height=1920):
    h, w = panel.shape[:2]

    # Scale ảnh chính theo chiều rộng để ảnh phủ ngang 1080
    scale = width / w
    new_w = width
    new_h = int(round(h * scale))

    resized = cv2.resize(panel, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    resized = sharpen(resized)

    # Nếu ảnh sau resize cao hơn 1920 thì crop nhẹ giữa
    if new_h >= height:
        y1 = (new_h - height) // 2
        final = resized[y1:y1 + height, :]
        return final, {
            "mode": "resize_width_crop_height",
            "placed_width": new_w,
            "placed_height": new_h,
            "crop_y1": int(y1),
            "crop_y2": int(y1 + height)
        }

    # Nếu còn thiếu chiều cao thì kéo dài nền trên/dưới từ chính ảnh gốc
    top_need = (height - new_h) // 2
    bottom_need = height - new_h - top_need

    top_strip = resized[0:1, :, :]
    bottom_strip = resized[-1:, :, :]

    top_ext = np.repeat(top_strip, top_need, axis=0)
    bottom_ext = np.repeat(bottom_strip, bottom_need, axis=0)

    final = np.vstack([top_ext, resized, bottom_ext])

    return final, {
        "mode": "extend_edges_no_ai",
        "placed_width": new_w,
        "placed_height": new_h,
        "top_extend": int(top_need),
        "bottom_extend": int(bottom_need)
    }


@app.post("/split-storyboard")
async def split_storyboard(
    file: UploadFile = File(...),
    rows: int = Query(4),
    cols: int = Query(2),
    width: int = Query(1080),
    height: int = Query(1920),
    margin: int = Query(0),
    quality: int = Query(96)
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

        for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
            panel = img[y1:y2, x1:x2]

            if panel.size == 0:
                continue

            final_img, debug = extend_to_9x16(panel, width, height)

            filename = f"{batch_id}_scene_{i:03}_extend_9x16.jpg"
            path = os.path.join(OUTPUT_DIR, filename)

            cv2.imwrite(path, final_img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])

            scenes.append({
                "scene": int(i),
                "fileName": filename,
                "mimeType": "image/jpeg",
                "width": int(width),
                "height": int(height),
                "ratio": "9:16",
                "url": f"{BASE_URL}/files/{filename}",
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
            "rows": int(rows),
            "cols": int(cols),
            "width": int(width),
            "height": int(height),
            "ratio": "9:16",
            "mode": "extend_edges_no_ai",
            "ai": "disabled",
            "scenes": scenes
        }

    except Exception as e:
        return JSONResponse(
            {"error": "Internal Server Error", "detail": str(e)},
            status_code=500
        )
