from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import os
import math
import uuid
import zipfile
import shutil
import numpy as np

app = FastAPI()


def merge_lines(lines, min_distance):
    merged = []

    for line in lines:
        if not merged or abs(line - merged[-1]) > min_distance:
            merged.append(line)

    return merged


def detect_grid(input_path):
    img = Image.open(input_path).convert("RGB")
    arr = np.array(img)

    height, width, _ = arr.shape

    col_scores = []
    for x in range(width):
        col_scores.append(np.std(arr[:, x, :]))

    threshold_col = np.percentile(col_scores, 8)

    vertical_lines = []
    in_line = False
    start = 0

    for i, score in enumerate(col_scores):
        if score <= threshold_col:
            if not in_line:
                start = i
                in_line = True
        else:
            if in_line:
                end = i
                if end - start > 2:
                    vertical_lines.append((start + end) // 2)
                in_line = False

    vertical_lines = [
        x for x in vertical_lines
        if width * 0.05 < x < width * 0.95
    ]

    vertical_lines = merge_lines(
        vertical_lines,
        min_distance=int(width * 0.05)
    )

    row_scores = []
    for y in range(height):
        row_scores.append(np.std(arr[y, :, :]))

    threshold_row = np.percentile(row_scores, 8)

    horizontal_lines = []
    in_line = False
    start = 0

    for i, score in enumerate(row_scores):
        if score <= threshold_row:
            if not in_line:
                start = i
                in_line = True
        else:
            if in_line:
                end = i
                if end - start > 2:
                    horizontal_lines.append((start + end) // 2)
                in_line = False

    horizontal_lines = [
        y for y in horizontal_lines
        if height * 0.05 < y < height * 0.95
    ]

    horizontal_lines = merge_lines(
        horizontal_lines,
        min_distance=int(height * 0.05)
    )

    cols = len(vertical_lines) + 1
    rows = len(horizontal_lines) + 1

    if cols < 1:
        cols = 1

    if rows < 1:
        rows = 1

    return cols, rows


def crop_grid_image(
    input_path,
    output_dir,
    cols=None,
    rows=None,
    total_frames=None,
    auto_detect=True
):
    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(input_path).convert("RGB")
    img_width, img_height = img.size

    if auto_detect and (cols is None or rows is None):
        detected_cols, detected_rows = detect_grid(input_path)

        if cols is None:
            cols = detected_cols

        if rows is None:
            rows = detected_rows

    if cols is None:
        cols = 3

    if rows is None:
        if total_frames:
            rows = math.ceil(total_frames / cols)
        else:
            rows = 1

    if total_frames is None:
        total_frames = cols * rows

    if cols <= 0 or rows <= 0:
        raise HTTPException(
            status_code=400,
            detail="cols và rows phải lớn hơn 0"
        )

    cell_width = img_width // cols
    cell_height = img_height // rows

    output_files = []
    frame_index = 0

    for r in range(rows):
        for c in range(cols):
            if frame_index >= total_frames:
                break

            left = c * cell_width
            top = r * cell_height
            right = img_width if c == cols - 1 else left + cell_width
            bottom = img_height if r == rows - 1 else top + cell_height

            cropped = img.crop((left, top, right, bottom))

            output_path = os.path.join(
                output_dir,
                f"frame_{frame_index + 1:02d}.jpg"
            )

            cropped.save(output_path, "JPEG", quality=95)
            output_files.append(output_path)

            frame_index += 1

    return output_files, cols, rows


def create_zip(files, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in files:
            zipf.write(file_path, arcname=os.path.basename(file_path))

    return zip_path


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Grid Crop API is running"
    }


@app.post("/crop")
async def crop_image(
    file: UploadFile = File(...),

    # để trống cols/rows thì API tự nhận diện
    cols: int | None = Form(None),
    rows: int | None = Form(None),

    # nếu biết tổng số frame thì truyền vào
    total_frames: int | None = Form(None),

    # bật/tắt auto detect
    auto_detect: bool = Form(True),
):
    job_id = str(uuid.uuid4())

    work_dir = f"/tmp/{job_id}"
    input_dir = os.path.join(work_dir, "input")
    output_dir = os.path.join(work_dir, "frames")
    zip_path = os.path.join(work_dir, "frames.zip")

    os.makedirs(input_dir, exist_ok=True)

    input_path = os.path.join(input_dir, file.filename)

    try:
        with open(input_path, "wb") as f:
            f.write(await file.read())

        frames, detected_cols, detected_rows = crop_grid_image(
            input_path=input_path,
            output_dir=output_dir,
            cols=cols,
            rows=rows,
            total_frames=total_frames,
            auto_detect=auto_detect
        )

        create_zip(frames, zip_path)

        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=f"frames_{detected_cols}x{detected_rows}.zip",
            headers={
                "X-Detected-Cols": str(detected_cols),
                "X-Detected-Rows": str(detected_rows),
                "X-Total-Frames": str(len(frames)),
            }
        )

    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))
