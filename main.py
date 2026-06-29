from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from PIL import Image, ImageFilter
import os
import uuid
import zipfile
import shutil

app = FastAPI()


def make_9_16_crop(img, target_width=1080, target_height=1920):
    src_w, src_h = img.size
    target_ratio = target_width / target_height
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        left = (src_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, src_h))
    else:
        new_h = int(src_w / target_ratio)
        top = (src_h - new_h) // 2
        img = img.crop((0, top, src_w, top + new_h))

    return img.resize((target_width, target_height), Image.LANCZOS)


def make_9_16_padding(img, target_width=1080, target_height=1920):
    bg = img.resize((target_width, target_height), Image.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(35))

    src_w, src_h = img.size
    scale = min(target_width / src_w, target_height / src_h)

    new_w = int(src_w * scale)
    new_h = int(src_h * scale)

    foreground = img.resize((new_w, new_h), Image.LANCZOS)

    x = (target_width - new_w) // 2
    y = (target_height - new_h) // 2

    bg.paste(foreground, (x, y))
    return bg


def crop_grid_image(
    input_path,
    output_dir,
    cols=3,
    rows=4,
    total_frames=11,
    crop_margin=0,
    fit_mode="crop"
):
    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(input_path).convert("RGB")
    img_width, img_height = img.size

    cell_width = img_width / cols
    cell_height = img_height / rows

    output_files = []
    frame_index = 0

    for r in range(rows):
        for c in range(cols):
            if frame_index >= total_frames:
                break

            left = round(c * cell_width) + crop_margin
            top = round(r * cell_height) + crop_margin
            right = round((c + 1) * cell_width) - crop_margin
            bottom = round((r + 1) * cell_height) - crop_margin

            if right <= left or bottom <= top:
                raise ValueError("crop_margin quá lớn, ảnh bị cắt sai")

            cropped = img.crop((left, top, right, bottom))

            if fit_mode == "padding":
                cropped = make_9_16_padding(cropped)
            else:
                cropped = make_9_16_crop(cropped)

            output_path = os.path.join(
                output_dir,
                f"frame_{frame_index + 1:02d}.jpg"
            )

            cropped.save(output_path, "JPEG", quality=95)
            output_files.append(output_path)

            frame_index += 1

    return output_files


def create_zip(files, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in files:
            zipf.write(file_path, arcname=os.path.basename(file_path))


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Storyboard Splitter API is running",
        "endpoint": "/crop"
    }


@app.post("/crop")
async def crop_image(
    file: UploadFile = File(...),
    cols: int = Form(3),
    rows: int = Form(4),
    total_frames: int = Form(11),
    crop_margin: int = Form(0),
    fit_mode: str = Form("crop")
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

        fit_mode = fit_mode.lower().strip()

        if fit_mode not in ["crop", "padding"]:
            raise HTTPException(
                status_code=400,
                detail="fit_mode chỉ nhận crop hoặc padding"
            )

        frames = crop_grid_image(
            input_path=input_path,
            output_dir=output_dir,
            cols=cols,
            rows=rows,
            total_frames=total_frames,
            crop_margin=crop_margin,
            fit_mode=fit_mode
        )

        create_zip(frames, zip_path)

        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename="frames_9x16.zip"
        )

    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))
