# # dataset.py
# """
# Dataset class for loading remote sensing images (e.g., multispectral and panchromatic).
#
# This module provides a PyTorch Dataset implementation to load TIFF images from a directory,
# """
#
import os
import logging
import numpy as np
from PIL import Image, ImageOps
import torch.utils.data as data
from osgeo import gdal
from args import config

# Configure logging
logger = logging.getLogger(__name__)

def is_tiff_image(filename):
    return filename.endswith('.tif')

def load_image(filepath):

    if not os.path.isfile(filepath):
        logger.error(f"Image file not found: {filepath}")
        raise FileNotFoundError(f"Image file not found: {filepath}")

    try:
        img = gdal.Open(filepath)
        if img is None:
            raise RuntimeError(f"Failed to open image: {filepath}")

        img_array = img.ReadAsArray()  # Shape: [C, H, W]
        if img_array.ndim == 3 and img_array.shape[0] in {1, 4}:
            img_array = img_array.transpose(1, 2, 0)  # To [H, W, C]
        img_array = img_array.astype(np.float32) / config.max_value
        logger.debug(f"Loaded image {filepath} with shape {img_array.shape}")
        return img_array
    except Exception as e:
        logger.error(f"Error loading image {filepath}: {e}")
        raise RuntimeError(f"Failed to load image: {e}")


def denormalize_image(img_array):

    return (img_array * config.max_value).astype(np.uint16)


class RemoteSensingDataset(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        if not os.path.isdir(data_dir):
            logger.error(f"Data directory not found: {data_dir}")
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        self.data_dir = data_dir
        self.transform = transform

        tiff_files = [x for x in os.listdir(data_dir) if is_tiff_image(x)]

        prefixes = []
        seen_prefixes = set()
        for filename in tiff_files:
            prefix = filename.split('_')[0]
            if prefix not in seen_prefixes:
                prefixes.append(prefix)
                seen_prefixes.add(prefix)

        self.image_prefixes = [os.path.join(data_dir, prefix) for prefix in prefixes]
        logger.info(f"Initialized dataset with {len(self.image_prefixes)} images from {data_dir}")


    def __getitem__(self, index):

        prefix = self.image_prefixes[index]
        try:
            pan = load_image(f"{prefix}_pan.tif")
            lr = load_image(f"{prefix}_lr.tif")
            lr_u = load_image(f"{prefix}_lr_u.tif")
            ms = load_image(f"{prefix}_mul.tif")

            if self.transform:
                pan = self.transform(pan)
                lr = self.transform(lr)
                lr_u = self.transform(lr_u)
                ms = self.transform(ms)

            return pan, lr, lr_u, ms
        except Exception as e:
            logger.error(f"Failed to load item {index} from {prefix}: {e}")
            raise RuntimeError(f"Failed to load item: {e}")

    def __len__(self):

        return len(self.image_prefixes)

