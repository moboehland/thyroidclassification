from pathlib import Path
import numpy as np
from PIL import Image
import imageio


def find_corresponding_labels(images_paths, labels_folder, labels_appendix):
    labels_paths = []
    labels_folder = Path(labels_folder)
    for image_path in images_paths:
        filename = image_path.stem + labels_appendix
        label_path = labels_folder.joinpath(filename)
        if not label_path.is_file():
            raise Exception(f"No corresponding label for filename {image_path.stem} found")
        labels_paths.append(label_path)
    return labels_paths


def min_max_normalization(img, min_value=None, max_value=None):
    """

    :param img:
    :param min_value:
    :param max_value:
    :return:
    """

    if max_value is None:
        max_value = img.max()

    if min_value is None:  # Get new minimum value

        min_value = img.min()

    # Clip image to filter hot and cold pixels
    img = np.clip(img, min_value, max_value)

    # Apply Min-max-normalization
    img = 2 * (img.astype(np.float32) - min_value) / (max_value - min_value) - 1

    return img.astype(np.float32)


def save_image(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, image)
