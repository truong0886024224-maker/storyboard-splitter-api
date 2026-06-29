from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import cv2, numpy as np, os, uuid, base64

app = FastAPI()
OUTPUT_DIR = "files"
BASE_URL = "https://storyboard-splitter-api.onrender.com"

os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="files")

@app.get("/")
def home():
    return {"status": "ok", "message": "Simple Storyboard Splitter"}

def make_grid_boxes(img, rows, cols, margin=2):
    h, w = img.shape[:2]
    boxes = []
    cell_w, cell_h = w / cols, h / rows

    for r in range(rows):
        for c in range(cols):
            boxes.append({
                "x1": max(0, int(round(c * cell_w)) + margin),
                "y1": max(0, int(round(r * cell_h)) + margin),
                "x2": min(w, int(round((c + 1) * cell_w)) - margin),
                "y2": min(h, int(round((r + 1) * cell_h)) - margin),
            })
    return boxes

def encode_base64(img, quality=96):
    ok, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode("utf-8") if ok else None

@app.post("/split-storyboard")
async def split_storyboard(
    file: UploadFile = File(...),
    rows: int = Query(4),
    cols: int = Query(2),
    margin: int = Query(2),
    quality: int = Query(96)
):
    try:
        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse({"error": "Cannot read image"}, status_code=400)

        boxes = make_grid_boxes(img, rows, cols, margin)
        batch_id = str(uuid.uuid4())[:8]
        scenes = []

        for i, box in enumerate(boxes, start=1):
            crop = img[box["y1"]:box["y2"], box["x1"]:box["x2"]]
            if crop.size == 0:
                continue

            filename = f"{batch_id}_scene_{i:03}.jpg"
            path = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, quality])

            h, w = crop.shape[:2]

            scenes.append({
                "scene": i,
                "fileName": filename,
                "mimeType": "image/jpeg",
                "width": w,
                "height": h,
                "url": f"{BASE_URL}/files/{filename}",
                "base64": encode_base64(crop, quality),
                "storyboard_box": box
            })

        return {
            "total": len(scenes),
            "rows": rows,
            "cols": cols,
            "mode": "simple_split_original",
            "scenes": scenes
        }

    except Exception as e:
        return JSONResponse({"error": "Internal Server Error", "detail": str(e)}, status_code=500)
