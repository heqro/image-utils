import numpy as np

try:
    import torch
    from torch import nonzero, Tensor
    import torch.nn.functional as F
except ImportError:
    torch = None

    class Tensor:
        pass


def get_bounding_box(
    mask: Tensor | np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Given a boolean 2D tensor mask, calculates the bounding box coordinates
    that encapsulate all True values in the mask.

    Args:
        mask (Tensor | np.ndarray): Binary 2D tensor with True values indicating foreground

    Returns:
        tuple: Tuple of tuples ((min_row, min_col), (max_row, max_col))
            containing the top-left and bottom-right coordinates of the box
    """
    if isinstance(mask, Tensor):
        indices = nonzero(mask, as_tuple=True)
        top_left = (
            int(indices[-2].min().item()),
            int(indices[-1].min().item()),
        )  # (min_row, min_col)
        bottom_right = (
            int(indices[-2].max().item()),
            int(indices[-1].max().item()),
        )  # (max_row, max_col)
        return top_left, bottom_right
    else:
        indices = np.argwhere(mask)
        min_row, min_col = indices.min(axis=0)
        max_row, max_col = indices.max(axis=0)

        return (min_row, min_col), (max_row, max_col)


def crop_image(img: Tensor | np.ndarray, d=32):
    """Crop the input image tensor to make the height and width divisible by d.

    Args:
        img (Tensor | np.ndarray): Input image tensor
        d (int): Divisor to make dimensions divisible by. Default is 32.

    Returns:
        Tensor: Cropped image tensor with dimensions divisible by d
    """
    new_height = img.shape[-2] - img.shape[-2] % d
    new_width = img.shape[-1] - img.shape[-1] % d
    return img[..., :new_height, :new_width]


def pad_image(img: Tensor | np.ndarray, new_height: int, new_width: int):
    """Pad image tensor along height and width dimensions using zeros to match target sizes.

    Args:
        img (Tensor | np.ndarray): Input image tensor of shape (channels, height, width)
        new_height (int): Target height to pad to
        new_width (int): Target width to pad to

    Returns:
        Tensor: Padded image tensor of shape (channels, new_height, new_width)
    """
    channels, img_height, img_width = img.shape
    if isinstance(img, Tensor):
        padding = torch.zeros(channels, new_height, new_width)
    else:
        padding = np.zeros((channels, new_height, new_width), dtype=img.dtype)
    padding[:, :img_height, :img_width] = img
    return padding
