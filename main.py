from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import os

from config import OUTPUT_DIR, BASE_URL, DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_QUALITY, DEFAULT_BACKGROUND
from detector import detect_panels
from canvas import fit_to_canvas
from utils import ensure_dir, make_batch_id, save_jpg, encode_jpg_base64

app = FastAPI()

ensure_dir(OUTPUT_DIR)
app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="files")


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Storyboard Splitter Pro is running"
    }


@app.post("/split-storyboard")
async def split_storyboard(
    file: UploadFile = File(...),
    width: int = Query(DEFAULT_WIDTH),
    height: int = Query(DEFAULT_HEIGHT),
    quality: int = Query(DEFAULT_QUALITY),
    bg: str = Query(DEFAULT_BACKGROUND),
    debug: bool = Query(False)
):
    try:
        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse({"error": "Cannot read image"}, status_code=400)

        boxes = detect_panels(img)

        if not boxes:
            return JSONResponse({
                "error": "No panels detected",
                "hint": "Storyboard cần có các ô tách biệt rõ bằng viền trắng hoặc khoảng trống."
            }, status_code=400)

        batch_id = make_batch_id()
        scenes = []

        debug_img = img.copy()

        for i, box in enumerate(boxes, start=1):
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

            panel = img[y1:y2, x1:x2]

            if panel.size == 0:
                continue

            final_img = fit_to_canvas(
                panel,
                width=width,
                height=height,
                background=bg
            )

            filename = f"{batch_id}_scene_{i:03}_9x16.jpg"
            save_jpg(final_img, OUTPUT_DIR, filename, quality)

            scenes.append({
                "scene": i,
                "fileName": filename,
                "mimeType": "image/jpeg",
                "width": width,
                "height": height,
                "ratio": "9:16",
                "url": f"{BASE_URL}/files/{filename}",
                "base64": encode_jpg_base64(final_img, quality),
                "box": box
            })

            if debug:
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(
                    debug_img,
                    str(i),
                    (x1 + 10, y1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3
                )

        response = {
            "total": len(scenes),
            "width": width,
            "height": height,
            "ratio": "9:16",
            "mode": "auto_contour_detection",
            "ai": "disabled",
            "background": bg,
            "scenes": scenes
        }

        if debug:
            debug_filename = f"{batch_id}_debug_boxes.jpg"
            save_jpg(debug_img, OUTPUT_DIR, debug_filename, quality)
            response["debug_url"] = f"{BASE_URL}/files/{debug_filename}"

        return response

    except Exception as e:
        return JSONResponse({
            "error": "Internal Server Error",
            "detail": str(e)
        }, status_code=500)
