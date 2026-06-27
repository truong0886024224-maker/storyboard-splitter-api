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
        "message": "Storyboard Splitter API 9x16 Blur Background Fixed is running"
    }


def fallback_grid(img, rows, cols):
    h, w = img.shape[:2]
    boxes = []

    cell_w = w // cols
    cell_h = h // rows

    for r in range(rows):
        for c in range(cols):
            x1 = c * cell_w + 4
            y1 = r * cell_h + 4
            x2 = (c + 1) * cell_w - 4 if c < cols - 1 else w - 4
            y2 = (r + 1) * cell_h - 4 if r < rows - 1 else h - 4

            boxes.append((
                max(0, int(x1)),
                max(0, int(y1)),
                min(w, int(x2)),
                min(h, int(y2))
            ))

    return boxes


def border_score(img, rows, cols):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    scores = []

    for c in range(1, cols):
        x = int(w * c / cols)
        band = gray[:, max(0, x - 4):min(w, x + 4)]
        scores.append(float(np.mean(band > 225)))

    for r in range(1, rows):
        y = int(h * r / rows)
        band = gray[max(0, y - 4):min(h, y + 4), :]
        scores.append(float(np.mean(band > 225)))

    return float(np.mean(scores)) if scores else 0.0


def choose_auto_layout(img):
    candidates = [
        (4, 2),
        (3, 2),
        (3, 3),
        (4, 3),
        (5, 2),
        (2, 2),
        (2, 3),
        (5, 3),
    ]

    best = (4, 2)
    best_score = -999

    for rows, cols in candidates:
        score = border_score(img, rows, cols)

        if score > best_score:
            best_score = score
            best = (rows, cols)

    return best


def detect_layout(img, rows=0, cols=0):
    if rows > 0 and cols > 0:
        return rows, cols

    return choose_auto_layout(img)


def make_9x16_blur_background(img, width=1080, height=1920):
    h, w = img.shape[:2]

    if h <= 0 or w <= 0:
        raise ValueError("Invalid image size")

    # 1. Nền blur phủ kín 9:16, không bị thiếu chiều cao/rộng
    scale_bg = max(width / w, height / h)
    bg_w = max(width, int(round(w * scale_bg)) + 4)
    bg_h = max(height, int(round(h * scale_bg)) + 4)

    bg = cv2.resize(
        img,
        (bg_w, bg_h),
        interpolation=cv2.INTER_CUBIC
    )

    x_bg = max(0, (bg_w - width) // 2)
    y_bg = max(0, (bg_h - height) // 2)

    bg = bg[y_bg:y_bg + height, x_bg:x_bg + width]

    # Fix chắc chắn đúng size 1080x1920
    if bg.shape[0] != height or bg.shape[1] != width:
        bg = cv2.resize(bg, (width, height), interpolation=cv2.INTER_CUBIC)

    bg = cv2.GaussianBlur(bg, (0, 0), 32)
    bg = cv2.addWeighted(bg, 0.75, np.zeros_like(bg), 0.25, 0)

    # 2. Ảnh chính giữ nguyên tỉ lệ, không crop, không vỡ bố cục
    scale_fg = min(width / w, height / h)
    fg_w = max(1, int(round(w * scale_fg)))
    fg_h = max(1, int(round(h * scale_fg)))

    fg = cv2.resize(
        img,
        (fg_w, fg_h),
        interpolation=cv2.INTER_LANCZOS4
    )

    # Sharpen ảnh chính
    blur = cv2.GaussianBlur(fg, (0, 0), 1.0)
    fg = cv2.addWeighted(fg, 1.4, blur, -0.4, 0)

    x = max(0, (width - fg_w) // 2)
    y = max(0, (height - fg_h) // 2)

    canvas = bg.copy()

    # Fix nếu fg sát biên
    paste_w = min(fg_w, width - x)
    paste_h = min(fg_h, height - y)

    canvas[y:y + paste_h, x:x + paste_w] = fg[:paste_h, :paste_w]

    return canvas, {
        "mode": "9x16_blur_background_keep_original_fixed",
        "original_width": int(w),
        "original_height": int(h),
        "foreground_width": int(fg_w),
        "foreground_height": int(fg_h),
        "x": int(x),
        "y": int(y)
    }


@app.post("/split-storyboard")
async def split_storyboard(
    file: UploadFile = File(...),
    rows: int = Query(0),
    cols: int = Query(0),
    width: int = Query(1080),
    height: int = Query(1920)
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

        final_rows, final_cols = detect_layout(img, rows, cols)
        boxes = fallback_grid(img, final_rows, final_cols)

        batch_id = str(uuid.uuid4())[:8]
        scenes = []

        for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
            frame = img[y1:y2, x1:x2]

            if frame.size == 0:
                continue

            final_img, debug = make_9x16_blur_background(
                frame,
                width,
                height
            )

            filename = f"{batch_id}_scene_{i:03}_9x16_{width}x{height}.jpg"
            path = os.path.join(OUTPUT_DIR, filename)

            cv2.imwrite(path, final_img, [cv2.IMWRITE_JPEG_QUALITY, 96])

            scenes.append({
                "scene": int(i),
                "fileName": filename,
                "mimeType": "image/jpeg",
                "width": int(width),
                "height": int(height),
                "ratio": "9:16",
                "url": f"{BASE_URL}/files/{filename}",
                "layout": {
                    "rows": int(final_rows),
                    "cols": int(final_cols)
                },
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
            "layout": {
                "rows": int(final_rows),
                "cols": int(final_cols)
            },
            "width": int(width),
            "height": int(height),
            "ratio": "9:16",
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
