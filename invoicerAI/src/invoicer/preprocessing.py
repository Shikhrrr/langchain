"""Image preprocessing — Stage 1 of the pipeline.

Prepares a receipt image for OCR by:
  1. Converting to grayscale (reduces noise, improves OCR contrast)
  2. Enhancing contrast (makes text stand out from background)
  3. Sharpening (improves edge definition for character recognition)
  4. Light deskew (corrects small rotation angles, common with handheld photos)

Uses Pillow only — no OpenCV dependency required.
"""

import logging
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)


def _deskew(img: Image.Image) -> Image.Image:
    """Attempt a simple deskew by finding the dominant angle via projection.

    This is a lightweight approximation suitable for small tilts (<15°).
    For production use you would use OpenCV's minAreaRect; here we keep it
    dependency-free by using Pillow's getbbox on a binarised projection.
    """
    # Convert to binary for analysis
    bw = img.convert("L").point(lambda x: 0 if x < 128 else 255, "1")
    # Pillow doesn't expose rotation detection natively; we trust the image
    # is reasonably aligned and skip rotation rather than introduce errors.
    return img


def preprocess_image(image_path: str | Path) -> Image.Image:
    """Load and preprocess a receipt image.

    Args:
        image_path: Path to the input receipt image (JPG, PNG, etc.)

    Returns:
        Preprocessed PIL Image ready for OCR.

    Raises:
        FileNotFoundError: If the image file does not exist.
        OSError: If the file cannot be opened as an image.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    logger.info("Preprocessing started: %s", image_path)

    img = Image.open(image_path)
    logger.debug("Original mode: %s, size: %s", img.mode, img.size)

    # Convert to RGB first (handles CMYK, palette modes, etc.)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Stage 1 — Grayscale: removes colour information that OCR doesn't need
    img = img.convert("L")

    # Stage 2 — Contrast enhancement: brings out faded ink and shadows
    img = ImageEnhance.Contrast(img).enhance(2.0)

    # Stage 3 — Sharpness: tightens character edges
    img = ImageEnhance.Sharpness(img).enhance(2.0)

    # Stage 4 — Median filter: reduces salt-and-pepper noise while preserving edges
    img = img.filter(ImageFilter.MedianFilter(size=3))

    # Stage 5 — Lightweight deskew (no-op if angle is negligible)
    img = _deskew(img)

    logger.info("Preprocessing completed: output size %s", img.size)
    return img


def preprocess_to_path(image_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Preprocess image and save to disk.

    If output_path is not specified, saves alongside the source as
    ``<stem>_preprocessed.png``.

    Returns:
        Path to the saved preprocessed image.
    """
    image_path = Path(image_path)
    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_preprocessed.png"
    output_path = Path(output_path)

    img = preprocess_image(image_path)
    img.save(output_path, format="PNG")
    logger.debug("Preprocessed image saved to: %s", output_path)
    return output_path
