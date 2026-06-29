from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import os
import uuid
import zipfile
import shutil

app = FastAPI()


def parse_row_layout(row_layout: str):
    try:
        layout = [int(x.strip()) for x in row_layout.split(",") if x.strip()]
    except Exception:
        raise HTTPException(status_code=400, detail="row_layout sai. Ví dụ đúng: 3,3,3,2")

    if not layout:
        raise HTTPException(status_code=400, detail="row_layout không được rỗng")

    if any(x <= 0 for x in layout):
        raise HTTPException(status_code=400, detail="Số cột mỗi hàng phải lớn hơn 0")

    return layout


def fit_to_9_16_no_crop(
    img: Image.Image,
    target_width: int = 1080,
    target_height: int = 1920,
    bg_color=(255, 255, 255)
):
    """
    Giữ toàn bộ chi tiết ảnh, không crop, không blur.
    Ảnh chính được fit vào khung 9:16, phần dư là nền trắng.
    """
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


def crop_storyboard_by_row_layout(
    input_path: str,
    output_dir: str,
    row_layout: list[int],
    crop_margin: int = 0,
    target_width: int = 1080,
    target_height: int = 1920,
    quality: int = 100,
    bg_color=(255, 255, 255)
):
    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(input_path).convert("RGB")
    img_width, img_height = img.size

    rows = len(row_layout)
    row_height = img_height / rows

    output_files = []
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

            frame_9_16 = fit_to_9_16_no_crop(
                frame,
                target_width=target_width,
                target_height=target_height,
                bg_color=bg_color
            )

            output_path = os.path.join(output_dir, f"frame_{frame_index:02d}.jpg")

            frame_9_16.save(
                output_path,
                "JPEG",
                quality=quality,
                optimize=True,
                progressive=True,
                subsampling=0
            )

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

    # Ví dụ ảnh 11 frame: 3 hàng đầu mỗi hàng 3 ảnh, hàng cuối 2 ảnh
    row_layout: str = Form("3,3,3,2"),

    # Để 0 để không mất chi tiết. Nếu dính viền thì tăng 2-4.
    crop_margin: int = Form(0),

    # Output chuẩn Shorts/Reels/TikTok
    target_width: int = Form(1080),
    target_height: int = Form(1920),

    # Chất lượng ảnh
    quality: int = Form(100),

    # Nền trắng. Nếu muốn nền đen đổi thành 0,0,0
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

        layout = parse_row_layout(row_layout)

        files = crop_storyboard_by_row_layout(
            input_path=input_path,
            output_dir=output_dir,
            row_layout=layout,
            crop_margin=crop_margin,
            target_width=target_width,
            target_height=target_height,
            quality=quality,
            bg_color=(bg_r, bg_g, bg_b)
        )

        create_zip(files, zip_path)

        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename="frames_9x16.zip"
        )

    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))
