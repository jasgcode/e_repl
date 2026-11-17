"""Images the model sees, encoded once when code hands them over.

``show()`` and sub-model queries take raw pixels and turn them into JPEG bytes here. The same
bytes go to the model and into the run's trajectory, so a logged run holds exactly what the model
was shown, after resizing and compression.

Pixels are BGR (or single-channel gray) ``uint8`` arrays: the order OpenCV and RealSense's
``bgr8`` stream produce.
"""

import base64
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

JPEG_MEDIA_TYPE = "image/jpeg"


@dataclass(frozen=True)
class ImageSettings:
    """How images are shrunk and compressed before the model sees them."""

    max_side: int = 768
    quality: int = 85


@dataclass(frozen=True)
class ObservedImage:
    jpeg: bytes
    caption: str
    width: int
    height: int

    def data_url(self) -> str:
        return f"data:{JPEG_MEDIA_TYPE};base64," + base64.b64encode(self.jpeg).decode("ascii")


def encode_image(pixels: np.ndarray, caption: str, settings: ImageSettings) -> ObservedImage:
    """Shrink ``pixels`` to fit ``settings.max_side`` and encode them as JPEG.

    Raises:
        TypeError: The pixels are not ``uint8``; the message tells the model how to fix it.
        ValueError: The array is not HxW or HxWx3.
    """
    if pixels.dtype != np.uint8:
        raise TypeError(f"images need uint8 pixels, got {pixels.dtype}; scale to 0-255 and cast first")
    if pixels.ndim == 3 and pixels.shape[2] == 3:
        pixels = pixels[..., ::-1]  # BGR to RGB, the order Pillow expects
    elif pixels.ndim != 2:
        raise ValueError(f"images must be HxW gray or HxWx3 BGR, got shape {pixels.shape}")
    image = Image.fromarray(np.ascontiguousarray(pixels))
    image.thumbnail((settings.max_side, settings.max_side))  # only ever shrinks
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=settings.quality)
    return ObservedImage(buffer.getvalue(), caption, image.width, image.height)
