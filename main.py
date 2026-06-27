from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64

app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok", "message": "Storyboard Splitter API is running"}

def make_9x16_full_hd(img, target_w=1080, target_h=1920):
    h, w = img.shape[:2]

    # Nền blur full 1080x1920
    scale_bg = max(target_w / w, target_h / h)
    bg_w = int(w * scale_bg)
    bg_h = int(h * scale_bg)
    bg = cv2.resize(img, (bg_w, bg_h), interpolation=cv2.INTER_CUBIC)

    x = (bg_w - target_w) // 2
    y = (bg_h - target_h) // 2
    bg = bg[y:y+target_h, x:x+target_w]
    bg = cv2.GaussianBlur(bg, (51, 51), 0)

    # Ảnh chính giữ nguyên tỷ lệ, nằm giữa
    scale_fg = min(target_w / w, target_h / h)
    fg_w = int(w * scale_fg)
    fg_h = int(h * scale_fg)
    fg = cv2.resize(img, (fg_w, fg_h), interpolation=cv2.INTER_CUBIC)

    canvas = bg.copy()
    x1 = (target_w - fg_w) // 2
    y1 = (target_h - fg_h) // 2
    canvas[y1:y1+fg_h, x1:x1+fg_w] = fg

    return canvas

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

            final_img = make_9x16_full_hd(crop)

            ok, buffer = cv2.imencode(".jpg", final_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            if ok:
                scenes.append({
                    "scene": index,
                    "fileName": f"scene_{index:03}_9x16_1080x1920.jpg",
                    "mimeType": "image/jpeg",
                    "width": 1080,
                    "height": 1920,
                    "base64": base64.b64encode(buffer).decode("utf-8")
                })

            index += 1

    return {"total": len(scenes), "scenes": scenes}
