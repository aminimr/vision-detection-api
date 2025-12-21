import cv2
import numpy as np
from PIL import Image
import io
from typing import Union, Tuple
from pathlib import Path


class ImageProcessor:

    @staticmethod
    def load_image(
            file: Union[str, Path, bytes, np.ndarray],
            target_size: Tuple[int, int] = None
    ) -> np.ndarray:
        try:
            if isinstance(file, (str, Path)):
                image = cv2.imread(str(file))
                if image is None:
                    raise ValueError(f"نمی‌توان تصویر را خواند: {file}")

            elif isinstance(file, bytes):
                nparr = np.frombuffer(file, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            elif isinstance(file, np.ndarray):
                image = file

            elif isinstance(file, Image.Image):
                image = cv2.cvtColor(np.array(file), cv2.COLOR_RGB2BGR)

            else:
                raise TypeError("فرمت فایل پشتیبانی نمی‌شود")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if target_size:
                image_rgb = cv2.resize(image_rgb, target_size)

            return image_rgb

        except Exception as e:
            raise ValueError(f"خطا در بارگذاری تصویر: {e}")

    @staticmethod
    def save_image(
            image: np.ndarray,
            path: Union[str, Path],
            quality: int = 95
    ) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image

        cv2.imwrite(str(path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return str(path)

    @staticmethod
    def validate_image_size(
            image: np.ndarray,
            max_size: Tuple[int, int] = (3840, 2160),  # 4K
            min_size: Tuple[int, int] = (64, 64)
    ) -> bool:
        height, width = image.shape[:2]

        if width > max_size[0] or height > max_size[1]:
            raise ValueError(f"تصویر بسیار بزرگ است. حداکثر سایز: {max_size}")

        if width < min_size[0] or height < min_size[1]:
            raise ValueError(f"تصویر بسیار کوچک است. حداقل سایز: {min_size}")

        return True

    @staticmethod
    def preprocess_image(
            image: np.ndarray,
            normalize: bool = True
    ) -> np.ndarray:
        if normalize:
            image = image.astype(np.float32) / 255.0

        return image


# Singleton instance
image_processor = ImageProcessor()