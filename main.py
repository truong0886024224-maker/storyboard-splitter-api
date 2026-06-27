from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import base64
import os
import uuid

app = FastAPI()

OUTPUT_DIR = "files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="files")


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Storyboard Splitter API is running"
    }


def save_image(img, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


@app.post("/split-storyboard")
async def split_storyboard(file: UploadFile = File(...)):
    contents = await file.read()

    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(
            {"error": "Cannot read image"},
            status_code=400
        )

    h, w = img.shape[:2]

    rows = 4
    cols = 2
    margin = 4

    scenes = []
    index = 1

    batch_id = str(uuid.uuid4())[:8]

    cell_w = w // cols
    cell_h = h // rows

    base_url = "https://storyboard-splitter-api.onrender.com"

    for r in range(rows):
        for c in range(cols):
            x1 = c * cell_w
            y1 = r * cell_h
            x2 = (c + 1) * cell_w if c < cols - 1 else w
            y2 = (r + 1) * cell_h if r < rows - 1 else h

            crop = img[y1:y2, x1:x2]

            if crop.shape[0] > margin * 2 and crop.shape[1] > margin * 2:
                crop = crop[margin:-margin, margin:-margin]

            filename = f"{batch_id}_scene_{index:03}.jpg"
            save_image(crop, filename)

            ok, buffer = cv2.imencode(
                ".jpg",
                crop,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )

            if ok:
                scenes.append({
                    "scene": index,
                    "fileName": filename,
                    "mimeType": "image/jpeg",
                    "url": f"{base_url}/files/{filename}",
                    "base64": base64.b64encode(buffer).decode("utf-8")
                })

            index += 1

    return {
        "total": len(scenes),
        "scenes": scenes
    }
