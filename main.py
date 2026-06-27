from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64

app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok", "message": "Storyboard Splitter API is running"}

@app.post("/split-storyboard")
async def split_storyboard(file: UploadFile = File(...)):
    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"error": "Cannot read image"}, status_code=400)

    h, w = img.shape[:2]

    rows = 4
    cols = 2
    margin = 4

    scenes = []
    index = 1

    cell_w = w // cols
    cell_h = h // rows

    for r in range(rows):
        for c in range(cols):
            x1 = c * cell_w
            y1 = r * cell_h
            x2 = (c + 1) * cell_w if c < cols - 1 else w
            y2 = (r + 1) * cell_h if r < rows - 1 else h

            crop = img[y1:y2, x1:x2]

            if crop.shape[0] > margin * 2 and crop.shape[1] > margin * 2:
                crop = crop[margin:-margin, margin:-margin]

            ok, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            if ok:
                scenes.append({
                    "scene": index,
                    "fileName": f"scene_{index:03}.jpg",
                    "mimeType": "image/jpeg",
                    "base64": base64.b64encode(buffer).decode("utf-8")
                })

            index += 1

    return {"total": len(scenes), "scenes": scenes}
