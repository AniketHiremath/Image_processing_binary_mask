import cv2
import numpy as np
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import logging

# logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_image(image_path):
    """
    Process a single image: read it, create a binary mask, save it, and return pixel count
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        logging.error(f"Failed to read image: {image_path}")
        return 0

    # Create binary mask where all channels > 200
    mask = np.all(img > 200, axis=2).astype(np.uint8) * 255

    # Create output path
    input_path = Path(image_path)
    output_path = input_path.parent / 'masks' / f'{input_path.stem}_mask.png'
    output_path.parent.mkdir(exist_ok=True)

    # Write mask
    cv2.imwrite(str(output_path), mask)

    # Count white pixels
    white_pixels = np.sum(mask == 255)

    logging.info(f"Processed {image_path}: {white_pixels} bright pixels")
    return white_pixels

def process_directory(input_dir):
    """
    Process all image files in a directory in parallel
    """
    # Get all supported image files
    input_path = Path(input_dir)
    image_files = []

    supported_extensions = {'.jpg', '.jpeg', '.png'}

    # Gather all image files with supported extensions
    for ext in supported_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))

    # Remove any duplicates
    image_files = list(set(image_files))

    if not image_files:
        logging.warning(f"No supported image files found in {input_dir}")
        return

    # Process images in parallel
    num_processes = min(cpu_count(), len(image_files))

    with Pool(num_processes) as pool:
        pixel_counts = pool.map(process_image, image_files)

    # Sum total bright pixels
    total_bright_pixels = sum(pixel_counts)
    logging.info(f"Total bright pixels across all images: {total_bright_pixels}")


if __name__ == '__main__':
    input_directory = './sample_images/'
    process_directory(input_directory)