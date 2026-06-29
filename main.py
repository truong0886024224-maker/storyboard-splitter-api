from PIL import Image
import os
import math


def crop_grid_image(
    input_path,
    output_dir="output_frames",
    cols=3,
    total_frames=None,
    rows=None,
    output_prefix="frame",
    output_format="jpg"
):
    """
    Cắt ảnh collage dạng lưới thành nhiều ảnh nhỏ.

    input_path: đường dẫn ảnh gốc
    output_dir: thư mục lưu ảnh sau khi cắt
    cols: số cột cố định, ví dụ 3
    total_frames: tổng số frame muốn lấy, ví dụ 11
    rows: số hàng nếu biết trước. Nếu không truyền, sẽ tự tính từ total_frames
    """

    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(input_path)
    img_width, img_height = img.size

    if rows is None:
        if total_frames is None:
            raise ValueError("Bạn cần truyền rows hoặc total_frames")
        rows = math.ceil(total_frames / cols)

    if total_frames is None:
        total_frames = rows * cols

    cell_width = img_width // cols
    cell_height = img_height // rows

    frame_index = 0

    for r in range(rows):
        for c in range(cols):
            if frame_index >= total_frames:
                break

            left = c * cell_width
            top = r * cell_height

            right = left + cell_width
            bottom = top + cell_height

            # tránh bị thiếu pixel ở cột/hàng cuối do chia dư
            if c == cols - 1:
                right = img_width
            if r == rows - 1:
                bottom = img_height

            cropped = img.crop((left, top, right, bottom))

            output_path = os.path.join(
                output_dir,
                f"{output_prefix}_{frame_index + 1:02d}.{output_format}"
            )

            if output_format.lower() in ["jpg", "jpeg"]:
                cropped = cropped.convert("RGB")
                cropped.save(output_path, quality=95)
            else:
                cropped.save(output_path)

            print(f"Saved: {output_path}")

            frame_index += 1

    print(f"Done. Total frames: {frame_index}")


# =========================
# CÁCH DÙNG
# =========================

# Ví dụ 1:
# Ảnh có 3 cột, tổng 11 frame.
# Code tự tính số hàng = ceil(11 / 3) = 4
crop_grid_image(
    input_path="grid.jpg",
    output_dir="frames",
    cols=3,
    total_frames=11
)


# Ví dụ 2:
# Nếu biết chắc ảnh là 3 cột x 4 hàng
# crop_grid_image(
#     input_path="grid.jpg",
#     output_dir="frames",
#     cols=3,
#     rows=4
# )
