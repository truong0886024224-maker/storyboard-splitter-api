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
    return {"status": "ok", "message": "Storyboard Splitter API is running"}


def crop_to_9x16(img):
    h, w = img.shape[:2]
    target_ratio = 9 / 16
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x1 = (w - new_w) // 2
        img = img[:, x1:x1 + new_w]
    else:
        new_h = int(w / target_ratio)
        y1 = max(0, (h - new_h) // 2)
        img = img[y1:y1 + new_h, :]

    return img


def enhance_image(img, target_w=1080, target_h=1920):
    img = crop_to_9x16(img)

    img = cv2.resize(
        img,
        (target_w, target_h),
        interpolation=cv2.INTER_LANCZOS4
    )

    # sharpen nhẹ, không làm giả ảnh quá mạnh
    blur = cv2.GaussianBlur(img, (0, 0), 1.2)
    sharp = cv2.addWeighted(img, 1.35, blur, -0.35, 0)

    return sharp


def detect_grid(img):
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # tìm đường trắng/ngăn cách giữa các frame
    vertical_score = np.mean(gray > 235, axis=0)
    horizontal_score = np.mean(gray > 235, axis=1)

    v_lines = np.where(vertical_score > 0.75)[0]
    h_lines = np.where(horizontal_score > 0.75)[0]

    def group_lines(lines):
        groups = []
        if len(lines) == 0:
            return groups

        start = lines[0]
        prev = lines[0]

        for x in lines[1:]:
            if x - prev > 3:
                groups.append((start, prev))
                start = x
            prev = x

        groups.append((start, prev))
        return groups

    v_groups = group_lines(v_lines)
    h_groups = group_lines(h_lines)

    v_centers = [int((a + b) / 2) for a, b in v_groups if b - a > 1]
    h_centers = [int((a + b) / 2) for a, b in h_groups if b - a > 1]

    xs = [0] + v_centers + [w]
    ys = [0] + h_centers + [h]

    xs = sorted(list(set(xs)))
    ys = sorted(list(set(ys)))

    boxes = []

    for r in range(len(ys) - 1):
        for c in range(len(xs) - 1):
            x1, x2 = xs[c], xs[c + 1]
            y1, y2 = ys[r], ys[r + 1]

            bw = x2 - x1
            bh = y2 - y1

            if bw < w * 0.12 or bh < h * 0.12:
                continue

            margin = 4
            boxes.append((
                max(0, x1 + margin),
                max(0, y1 + margin),
                min(w, x2 - margin),
                min(h, y2 - margin)
            ))

    return boxes


def fallback_grid(img, rows, cols):
    h, w = img.shape[:2]
    boxes = []

    cell_w = w // cols
    cell_h = h // rows

    for r in range(rows):
        for c in range(cols):
            x1 = c * cell_w
            y1 = r * cell_h
            x2 = (c + 1) * cell_w if c < cols - 1 else w
            y2 = (r + 1) * cell_h if r < rows - 1 else h

            margin = 4
            boxes.append((
                x1 + margin,
                y1 + margin,
                x2 - margin,
                y2 - margin
            ))

    return boxes


@app.post("/split-storyboard")
async def split_storyboard(
    file: UploadFile = File(...),
    rows: int = Query(0),
    cols: int = Query(0),
    width: int = Query(1080),
    height: int = Query(1920)
):
    contents = await file.read()

    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"error": "Cannot read image"}, status_code=400)

    if rows > 0 and cols > 0:
        boxes = fallback_grid(img, rows, cols)
    else:
        boxes = detect_grid(img)

        # nếu auto detect fail thì fallback phổ biến 4x3
        if len(boxes) < 4:
            boxes = fallback_grid(img, 4, 3)

    batch_id = str(uuid.uuid4())[:8]
    scenes = []

    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        final_img = enhance_image(crop, width, height)

        filename = f"{batch_id}_scene_{i:03}_9x16_{width}x{height}.jpg"
        path = os.path.join(OUTPUT_DIR, filename)

        cv2.imwrite(path, final_img, [cv2.IMWRITE_JPEG_QUALITY, 96])

        ok, buffer = cv2.imencode(
            ".jpg",
            final_img,
            [cv2.IMWRITE_JPEG_QUALITY, 96]
        )

        scenes.append({
            "scene": i,
            "fileName": filename,
            "mimeType": "image/jpeg",
            "width": width,
            "height": height,
            "url": f"{BASE_URL}/files/{filename}",
            "base64": base64.b64encode(buffer).decode("utf-8") if ok else None
        })

    return {
        "total": len(scenes),
        "width": width,
        "height": height,
        "ratio": "9:16",
        "scenes": scenes
    }
