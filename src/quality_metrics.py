from torch import Tensor
import torch
import numpy as np
def psnr_with_mask(
    img_1: Tensor | np.ndarray,
    img_2: Tensor | np.ndarray,
    mask: Tensor | np.ndarray,
    data_range=1.0,
):
    """
    Calculate PSNR between two images with a mask.

    Args:
        img_1 (Tensor | np.ndarray): First image
        img_2 (Tensor | np.ndarray): Second image
        mask (Tensor | np.ndarray): Mask to specify which pixels to use for calculation
        data_range (float): Maximum value range of the images (default: 1.0)

    Returns:
        float: PSNR value
    """
    mask_size = (mask > 0).sum().item()
    mse = ((img_1 - img_2) ** 2 * mask).sum() / mask_size
    if (
        isinstance(img_1, Tensor)
        and isinstance(img_2, Tensor)
        and isinstance(mask, Tensor)
    ):
        return 10 * torch.log10(data_range**2 / mse)
    else:
        return 10 * np.log10(data_range**2 / mse)
