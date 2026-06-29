from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image
import os
import uuid
import shutil
import base64
from io import BytesIO

app = FastAPI()


def parse_row_layout(row_layout: str):
    try:
        layout = [int(x.strip()) for x in row_layout.split(",") if x.strip()]
    except Exception:
        raise HTTPException(status_code=400, detail="row_layout sai. Ví dụ: 3,3,3,2")

    if not layout:
        raise HTTPException(status_code=400, detail="row_layout không được rỗng")

    if any(x <= 0 for x in layout):
        raise HTTPException(status_code=400, detail="Mỗi hàng phải có số ảnh > 0")

    return layout


def fit_to_9_16_no_crop(
    img: Image.Image,
    target_width=1080,
    target_height=1920,
    bg_color=(255, 255, 255)
):
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


def image_to_base64(img: Image.Image, quality=100):
    buffer = BytesIO()

    img.save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=True,
        subsampling=0
    )

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def split_storyboard(
    input_path: str,
    row_layout: list[int],
    crop_margin=0,
    target_width=1080,
    target_height=1920,
    quality=100,
    bg_color=(255, 255, 255)
):
    img = Image.open(input_path).convert("RGB")
    img_width, img_height = img.size

    rows = len(row_layout)
    row_height = img_height / rows

    frames = []
    frame_index = 1

    for row_index, cols_in_row in enumerate(row_layout):
        row_top = round(row_index * row_height)
        row_bottom = round((row_index + 1) * row_height)

        row_img = img.crop((0, row_top, img_width, row_bottom))
        col_width = img_width / cols_in_row

        for col_index in range(cols_in_row):
            left = round(col_index * col_width) + crop_margin
            right = round((col_index + 1) * col_width) - crop_margin
            top = crop_margin
            bottom = row_img.height - crop_margin

            if right <= left or bottom <= top:
                raise HTTPException(
                    status_code=400,
                    detail="crop_margin quá lớn, ảnh bị cắt sai"
                )

            frame = row_img.crop((left, top, right, bottom))

            final_img = fit_to_9_16_no_crop(
                frame,
                target_width=target_width,
                target_height=target_height,
                bg_color=bg_color
            )

            filename = f"frame_{frame_index:02d}.jpg"

            frames.append({
                "index": frame_index,
                "fileName": filename,
                "mimeType": "image/jpeg",
                "width": target_width,
                "height": target_height,
                "base64": image_to_base64(final_img, quality=quality)
            })

            frame_index += 1

    return frames


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Storyboard Splitter API is running",
        "endpoint": "/crop-json"
    }


@app.post("/crop-json")
async def crop_json(
    file: UploadFile = File(...),

    # Ví dụ storyboard 11 ảnh: 3 + 3 + 3 + 2
    row_layout: str = Form("3,3,3,2"),

    # Để 0 để không mất chi tiết
    crop_margin: int = Form(0),

    # Chuẩn 9:16
    target_width: int = Form(1080),
    target_height: int = Form(1920),

    # Ảnh sắc nét
    quality: int = Form(100),

    # Nền trắng
    bg_r: int = Form(255),
    bg_g: int = Form(255),
    bg_b: int = Form(255),
):
    job_id = str(uuid.uuid4())
    work_dir = f"/tmp/{job_id}"
    os.makedirs(work_dir, exist_ok=True)

    input_path = os.path.join(work_dir, file.filename)

    try:
        with open(input_path, "wb") as f:
            f.write(await file.read())

        layout = parse_row_layout(row_layout)

        frames = split_storyboard(
            input_path=input_path,
            row_layout=layout,
            crop_margin=crop_margin,
            target_width=target_width,
            target_height=target_height,
            quality=quality,
            bg_color=(bg_r, bg_g, bg_b)
        )

        return {
            "success": True,
            "total": len(frames),
            "row_layout": row_layout,
            "frames": frames
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
