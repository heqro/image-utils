from torch import nonzero, zeros, Tensor

def get_bounding_box(mask: Tensor) -> tuple[tuple[int, int], tuple[int, int]]:
    """Given a boolean 2D tensor mask, calculates the bounding box coordinates
    that encapsulate all True values in the mask.

    Args:
        mask (Tensor): Binary 2D tensor with True values indicating foreground

    Returns:
        tuple: Tuple of tuples ((min_row, min_col), (max_row, max_col))
            containing the top-left and bottom-right coordinates of the box
    """
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


def crop_image(img, d=32):
    """Crop the input image tensor to make the height and width divisible by d.

    Args:
        img (Tensor): Input image tensor
        d (int): Divisor to make dimensions divisible by. Default is 32.

    Returns:
        Tensor: Cropped image tensor with dimensions divisible by d
    """
    new_height = img.shape[-2] - img.shape[-2] % d
    new_width = img.shape[-1] - img.shape[-1] % d
    return img[..., :new_height, :new_width]


def pad_image(img: Tensor, new_height: int, new_width: int):
    """Pad image tensor along height and width dimensions using zeros to match target sizes.

    Args:
        img (Tensor): Input image tensor of shape (channels, height, width)
        new_height (int): Target height to pad to
        new_width (int): Target width to pad to

    Returns:
        Tensor: Padded image tensor of shape (channels, new_height, new_width)
    """
    channels = img.shape[0]
    padding = zeros(channels, new_height, new_width)
    _, img_height, img_width = img.shape
    padding[:, :img_height, :img_width] = img
    return padding
