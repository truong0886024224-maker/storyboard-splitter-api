from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import numpy as np
import cv2
import os
import uuid
import zipfile
import shutil

app = FastAPI()


def fit_9_16_no_crop(img, target_width=1080, target_height=1920, bg_color=(255, 255, 255)):
    img = img.convert("RGB")

    src_w, src_h = img.size
    scale = min(target_width / src_w, target_height / src_h)

    new_w = int(src_w * scale)
    new_h = int(src_h * scale)

    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (target_width, target_height), bg_color)

    x = (target_width - new_w) // 2
    y = (target_height - new_h) // 2

    canvas.paste(resized, (x, y))
    return canvas


def smooth(arr, kernel_size=7):
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(arr, kernel, mode="same")


def find_separator_bands(scores, threshold, min_size=1):
    bands = []
    in_band = False
    start = 0

    for i, value in enumerate(scores):
        if value >= threshold and not in_band:
            start = i
            in_band = True

        if value < threshold and in_band:
            end = i
            if end - start >= min_size:
                bands.append((start, end))
            in_band = False

    if in_band:
        end = len(scores)
        if end - start >= min_size:
            bands.append((start, end))

    return bands


def merge_bands(bands, max_gap=5):
    if not bands:
        return []

    merged = [bands[0]]

    for start, end in bands[1:]:
        prev_start, prev_end = merged[-1]

        if start - prev_end <= max_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    return merged


def remove_edge_bands(bands, size, edge_ratio=0.01):
    edge = int(size * edge_ratio)
    result = []

    for start, end in bands:
        center = (start + end) / 2

        if center <= edge:
            continue

        if center >= size - edge:
            continue

        result.append((start, end))

    return result


def segments_from_bands(size, bands, min_size=40):
    segments = []
    current = 0

    for start, end in bands:
        if start - current >= min_size:
            segments.append((current, start))
        current = end

    if size - current >= min_size:
        segments.append((current, size))

    return segments


def trim_dark_border(pil_img, threshold=8):
    arr = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    mask = gray > threshold

    if not mask.any():
        return pil_img

    ys, xs = np.where(mask)

    left = xs.min()
    right = xs.max() + 1
    top = ys.min()
    bottom = ys.max() + 1

    return pil_img.crop((left, top, right, bottom))


def detect_horizontal_bands(gray, dark_threshold=35):
    dark_mask = gray <= dark_threshold
    scores = dark_mask.mean(axis=1)
    scores = smooth(scores, 9)

    threshold = max(0.35, np.percentile(scores, 92))

    bands = find_separator_bands(scores, threshold, min_size=1)
    bands = merge_bands(bands, max_gap=6)
    bands = remove_edge_bands(bands, gray.shape[0], edge_ratio=0.01)

    return bands


def detect_vertical_bands(row_gray, dark_threshold=35):
    dark_mask = row_gray <= dark_threshold
    scores = dark_mask.mean(axis=0)
    scores = smooth(scores, 7)

    threshold = max(0.25, np.percentile(scores, 92))

    bands = find_separator_bands(scores, threshold, min_size=1)
    bands = merge_bands(bands, max_gap=6)
    bands = remove_edge_bands(bands, row_gray.shape[1], edge_ratio=0.01)

    return bands


def detect_panels(input_path, dark_threshold=35, crop_margin=0):
    pil_img = Image.open(input_path).convert("RGB")
    arr = np.array(pil_img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    img_h, img_w = gray.shape

    horizontal_bands = detect_horizontal_bands(gray, dark_threshold)
    row_segments = segments_from_bands(img_h, horizontal_bands, min_size=80)

    panels = []

    for row_top, row_bottom in row_segments:
        row_gray = gray[row_top:row_bottom, :]

        vertical_bands = detect_vertical_bands(row_gray, dark_threshold)
        col_segments = segments_from_bands(img_w, vertical_bands, min_size=80)

        for col_left, col_right in col_segments:
            left = col_left + crop_margin
            right = col_right - crop_margin
            top = row_top + crop_margin
            bottom = row_bottom - crop_margin

            if right <= left or bottom <= top:
                continue

            panel = pil_img.crop((left, top, right, bottom))
            panel = trim_dark_border(panel)

            if panel.width < 80 or panel.height < 80:
                continue

            panels.append(panel)

    return panels


def save_frames_to_zip(
    input_path,
    output_dir,
    target_width=1080,
    target_height=1920,
    quality=95,
    crop_margin=0,
    dark_threshold=35,
    bg_color=(255, 255, 255)
):
    os.makedirs(output_dir, exist_ok=True)

    panels = detect_panels(
        input_path=input_path,
        dark_threshold=dark_threshold,
        crop_margin=crop_margin
    )

    if len(panels) == 0:
        raise HTTPException(
            status_code=400,
            detail="Không detect được panel nào. Thử dark_threshold = 50 hoặc 70."
        )

    output_files = []

    for index, panel in enumerate(panels, start=1):
        frame = fit_9_16_no_crop(
            panel,
            target_width=target_width,
            target_height=target_height,
            bg_color=bg_color
        )

        output_path = os.path.join(output_dir, f"frame_{index:02d}.jpg")

        frame.save(
            output_path,
            "JPEG",
            quality=quality,
            optimize=True,
            progressive=True,
            subsampling=0
        )

        output_files.append(output_path)

    return output_files


def create_zip(files, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in files:
            zipf.write(file_path, arcname=os.path.basename(file_path))


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Storyboard Splitter ZIP API is running",
        "endpoint": "/crop"
    }


@app.post("/crop")
async def crop_zip(
    file: UploadFile = File(...),
    target_width: int = Form(1080),
    target_height: int = Form(1920),
    quality: int = Form(95),
    crop_margin: int = Form(0),
    dark_threshold: int = Form(35),
    bg_r: int = Form(255),
    bg_g: int = Form(255),
    bg_b: int = Form(255),
):
    job_id = str(uuid.uuid4())

    work_dir = f"/tmp/{job_id}"
    input_dir = os.path.join(work_dir, "input")
    output_dir = os.path.join(work_dir, "frames")
    zip_path = os.path.join(work_dir, "frames_9x16.zip")

    os.makedirs(input_dir, exist_ok=True)

    input_path = os.path.join(input_dir, file.filename)

    try:
        with open(input_path, "wb") as f:
            f.write(await file.read())

        files = save_frames_to_zip(
            input_path=input_path,
            output_dir=output_dir,
            target_width=target_width,
            target_height=target_height,
            quality=quality,
            crop_margin=crop_margin,
            dark_threshold=dark_threshold,
            bg_color=(bg_r, bg_g, bg_b)
        )

        create_zip(files, zip_path)

        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename="frames_9x16.zip",
            headers={
                "X-Total-Frames": str(len(files))
            }
        )

    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))
