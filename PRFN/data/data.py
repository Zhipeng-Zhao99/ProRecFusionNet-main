# data.py

import os
import logging
import numpy as np
import torch
from torchvision.transforms import Compose
from data.dataset  import RemoteSensingDataset


# Configure logging
logger = logging.getLogger(__name__)


class NumpyToTensor:

    def __call__(self, data):
        if not isinstance(data, np.ndarray):
            logger.error("Input to NumpyToTensor must be a numpy array")
            raise TypeError("Input must be a numpy array")

        if data.ndim == 3:
            # Transpose image data (H, W, C) to (C, H, W)
            data = np.transpose(data, (2, 0, 1))
            return torch.from_numpy(data).float()
        else:
            return torch.from_numpy(data).unsqueeze(0).float()


def get_data_transforms():
    return Compose([NumpyToTensor()])


def load_training_dataset(data_dir):

    if not os.path.isdir(data_dir):
        logger.error(f"Training data directory not found: {data_dir}")
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    logger.info(f"Loading training dataset from {data_dir}")
    return RemoteSensingDataset(data_dir, transform=get_data_transforms())


def load_validation_dataset(data_dir):

    if not os.path.isdir(data_dir):
        logger.error(f"Validation data directory not found: {data_dir}")
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    logger.info(f"Loading validation dataset from {data_dir}")
    return RemoteSensingDataset(data_dir, transform=get_data_transforms())


def load_testing_dataset(data_dir):

    if not os.path.isdir(data_dir):
        logger.error(f"Testing data directory not found: {data_dir}")
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    logger.info(f"Loading testing dataset from {data_dir}")
    return RemoteSensingDataset(data_dir, transform=get_data_transforms())

