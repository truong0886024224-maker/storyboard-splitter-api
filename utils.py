import os
import uuid
import base64
import cv2


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def make_batch_id() -> str:
    return str(uuid.uuid4())[:8]


def save_jpg(img, output_dir: str, filename: str, quality: int = 96):
    path = os.path.join(output_dir, filename)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return path


def encode_jpg_base64(img, quality: int = 96):
    ok, buffer = cv2.imencode(
        ".jpg",
        img,
        [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    )

    if not ok:
        return None

    return base64.b64encode(buffer).decode("utf-8")
