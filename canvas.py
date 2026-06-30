import cv2
import numpy as np


def make_background(img, width, height, mode="black"):
    if mode == "white":
        return np.ones((height, width, 3), dtype=np.uint8) * 255

    if mode == "edge":
        color = np.mean(img.reshape(-1, 3), axis=0).astype(np.uint8)
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        bg[:] = color
        return bg

    return np.zeros((height, width, 3), dtype=np.uint8)


def fit_to_canvas(
    img,
    width=1080,
    height=1920,
    background="black"
):
    """
    Giữ nguyên ảnh.
    Không crop.
    Không méo.
    """

    h, w = img.shape[:2]

    scale = min(
        width / w,
        height / h
    )

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_LANCZOS4
    )

    canvas = make_background(
        img,
        width,
        height,
        background
    )

    x = (width - new_w) // 2
    y = (height - new_h) // 2

    canvas[
        y:y + new_h,
        x:x + new_w
    ] = resized

    return canvas


def fit_cover(
    img,
    width=1080,
    height=1920
):
    """
    Full màn hình.
    Có crop.
    """

    h, w = img.shape[:2]

    scale = max(
        width / w,
        height / h
    )

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_LANCZOS4
    )

    x = (new_w - width) // 2
    y = (new_h - height) // 2

    return resized[
        y:y + height,
        x:x + width
    ]
