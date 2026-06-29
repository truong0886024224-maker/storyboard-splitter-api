from fastapi import FastAPI, UploadFile, File
from PIL import Image
import os
import math
import uuid

app = FastAPI()


def crop_grid_image(input_path, output_dir, cols=3, total_frames=None, rows=None):
    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(input_path)
    w, h = img.size

    if rows is None:
        if total_frames is None:
            raise ValueError("Cần truyền rows hoặc total_frames")
        rows = math.ceil(total_frames / cols)

    if total_frames is None:
        total_frames = rows * cols

    cell_w = w // cols
    cell_h = h // rows

    files = []
    index = 0

    for r in range(rows):
        for c in range(cols):
            if index >= total_frames:
                break

            left = c * cell_w
            top = r * cell_h
            right = w if c == cols - 1 else left + cell_w
            bottom = h if r == rows - 1 else top + cell_h

            cropped = img.crop((left, top, right, bottom)).convert("RGB")

            out_path = os.path.join(output_dir, f"frame_{index+1:02d}.jpg")
            cropped.save(out_path, quality=95)

            files.append(out_path)
            index += 1

    return files


@app.post("/crop")
async def crop_image(
    file: UploadFile = File(...),
    cols: int = 3,
    total_frames: int = 11
):
    job_id = str(uuid.uuid4())
    input_path = f"/tmp/{job_id}_{file.filename}"
    output_dir = f"/tmp/{job_id}_frames"

    with open(input_path, "wb") as f:
        f.write(await file.read())

    frames = crop_grid_image(
        input_path=input_path,
        output_dir=output_dir,
        cols=cols,
        total_frames=total_frames
    )

    return {
        "success": True,
        "total": len(frames),
        "frames": frames
    }
