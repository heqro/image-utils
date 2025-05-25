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


import torch
import torch.nn.functional as F


# Ensure shape is (1, 1, H, W)
def prepare(img):
    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.ndim == 3:
        img = img.unsqueeze(0)
    return img


def compute_epi_torch(denoised: Tensor, reference: Tensor):

    denoised = prepare(denoised)
    reference = prepare(reference)

    # Sobel filters
    sobel_x = (
        torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
            dtype=denoised.dtype,
            device=denoised.device,
        ).view(1, 1, 3, 3)
        / 8.0
    )

    sobel_y = sobel_x.transpose(2, 3)

    def gradient_magnitude(img):
        gx = F.conv2d(img, sobel_x, padding=1)
        gy = F.conv2d(img, sobel_y, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-8)

    grad_denoised = gradient_magnitude(denoised)
    grad_reference = gradient_magnitude(reference)

    numerator = torch.sum(grad_denoised * grad_reference)
    denominator = torch.sqrt(torch.sum(grad_denoised**2) * torch.sum(grad_reference**2))

    return (numerator / denominator).item() if denominator > 0 else 0.0


def compute_cnr_torch(img: Tensor, mask: Tensor):
    img, mask = prepare(img), prepare(mask)
    roi = img[mask > 0]
    outside_roi = img[mask == 0]

    mu1, mu2 = roi.mean(), outside_roi.mean()
    sigma_bkg = outside_roi.std()

    cnr = torch.abs(mu1 - mu2) / (sigma_bkg * 1.53 + 1e-8)
    return cnr.item()
