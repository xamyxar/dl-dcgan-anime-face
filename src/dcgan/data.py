from config import settings # assuming src added to python-system-path inside main runner (jupyter notebook)
from typing import Tuple, Optional, Any, Dict
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import logging
logger = logging.getLogger(__name__)

#----------------------------------------------------------------------------------------
def get_inference_transform(image_size: int = 64, mean: Optional[Tuple[float, float, float]] = None, std: Optional[Tuple[float, float, float]] = None)-> torchvision.transforms.Compose:
    """
    Return Inference_transform with optional normalization (to mean and std)
    Args:
        image_size: target image-size
        mean: mean for 3 channels (optional)
        std: standard deviation for 3 channels (optional)
    Return:
        Callable of type torchvision.transforms.Compose
    Raises:
        Exception: Re-raises any exception caught during processing after logging
    """
    try:
        image_size_plus=int(1.15*image_size)
        transforms_list = [
            torchvision.transforms.Resize(image_size_plus),  # resize min-dim to image_size_plus
            torchvision.transforms.CenterCrop(image_size),  # crop the center with image_size
            torchvision.transforms.ToTensor(),  # Scale to [0.0, 1.0] PyTorch Tensor (C x H x W)
        ]

        # Only apply normalization if both mean and std are provided
        if mean is not None and std is not None:
            transforms_list.append(torchvision.transforms.Normalize(mean, std))

        return torchvision.transforms.Compose(transforms_list)
    except Exception as e:
        logger.error(f"Error in get_inference_transform: {e}")
        raise

#----------------------------------------------------------------------------------------
def get_train_transform(image_size: int = 64, mean: Optional[Tuple[float, float, float]] = None, std: Optional[Tuple[float, float, float]] = None)-> torchvision.transforms.Compose:
    """
    Return training transform with optional normalization (to mean and std).
    Args:
        image_size: target image-size
        mean: mean for 3 channels (optional)
        std: standard deviation for 3 channels (optional)
    Return:
        Callable of type torchvision.transforms.Compose
    Raises:
        Exception: Re-raises any exception caught during processing after logging
    """
    try:
        image_size_plus=int(1.15*image_size)
        transforms_list = [
            torchvision.transforms.Resize(image_size_plus),  # resize min-dim to image_size_plus
            torchvision.transforms.CenterCrop(image_size),  # crop the center with image_size
            torchvision.transforms.RandomHorizontalFlip(0.5), #
            # Add other Transformation as data augmentation
            torchvision.transforms.ToTensor(),  # Scale to [0.0, 1.0] PyTorch Tensor (C x H x W)
        ]

        # Only apply normalization if both mean and std are provided
        if mean is not None and std is not None:
            transforms_list.append(torchvision.transforms.Normalize(mean, std))

        return torchvision.transforms.Compose(transforms_list)
    except Exception as e:
        logger.error(f"Error in get_transform: {e}")
        raise


def get_dataloader(cfg: Any)-> torch.utils.data.DataLoader:
    """
    Return train data loader
    Args:
        - cfg: Configuration object containing required parameters like:
            image_size: target image-size
            batch_size: batch size
    Return:
        torch.utils.data.DataLoader for train train-dataset
    Raises:
        Exception: Re-raises any exception caught during processing after logging
    """
    try:
        # using Tanh
        mean_, std_ = cfg.mean, cfg.std

        train_dataset_full = torchvision.datasets.ImageFolder(root=settings.DATA_DIR, transform=get_train_transform(cfg.image_size, mean_, std_))
        dataloader = DataLoader(train_dataset_full, batch_size=cfg.batch_size, shuffle=True)
        return dataloader
    except Exception as e:
        logger.error(f"Error in get_dataloader: {e}")
        raise

#----------------------------------------------------------------------------------------
def get_mean_std(loader: torch.utils.data.DataLoader)-> Tuple[torch.Tensor, torch.Tensor]:
    """
    Try to find out mean and std of original image itself, used in transform, and also can be helpful
    if we need to fill black area during transformation
    using loop on batch-size base is more practical for large data set
    Args:
        loader: dataset-loader
    Return:
        mean, std (both of type torch.Tensor)
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        # Sum of pixels and squared pixels for mean and std calculation
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0

        for data, _ in loader:
            # data shape: [batch_size, CH, H, W] or
            # Mean over batch, height, and width (dim 0, 2, 3) no Channel
            #results in Channel mean [B,C,H,W]
            channels_sum += torch.mean(data, dim=[0, 2, 3])
            channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3]) #
            num_batches += 1

        mean = channels_sum / num_batches
        # std = sqrt(E[X^2] - (E[X])^2)
        std = (channels_squared_sum / num_batches - mean**2)**0.5

        return mean, std
    except Exception as e:
        logger.error(f"Error in get_mean_std: {e}")
        raise

#----------------------------------------------------------------------------------------
#
def get_images_by_indices(indices, dataloader, batch_size):
    """
    Helper function to fetch images by their absolute dataset index
    Args:
        - indices: lis tof absolute indices
        - dataloader: dataloader of the dataset that contains the images
        - batch_size : batch-size in the dataloader
    """
    fetched_images = []
    target_batches = {idx // batch_size for idx in indices}

    batch_cache = {}
    for current_batch_idx, (images, *_) in enumerate(dataloader):
        if current_batch_idx in target_batches:
            batch_cache[current_batch_idx] = images
            if len(batch_cache) == len(target_batches):
                break

    for idx in indices:
        b_idx = idx // batch_size
        item_idx = idx % batch_size
        fetched_images.append(batch_cache[b_idx][item_idx])

    return torch.stack(fetched_images)