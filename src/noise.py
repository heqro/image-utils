from torch import Tensor, randn

def add_gaussian_noise(img: Tensor, std: float, avg: float = 0) -> Tensor:
    """Add Gaussian noise to a tensor.

    Args:
        img (Tensor): Image tensor to add noise to.
        std (float): Standard deviation of the noise.
        avg (float): Mean of the noise (default 0).

    Returns:
        Tensor: Image tensor with added Gaussian noise.
    """
    noise = avg + std * randn(img.shape, device=img.device)
    return img + noise


def add_rician_noise(img: Tensor, std: float) -> Tensor:
    """Add Rician noise to a tensor.

    Args:
        img (Tensor): Image tensor to add noise to.
        std (float): Standard deviation of the noise.

    Returns:
        Tensor: Image tensor with added Rician noise.
    """
    img_real = img + std * randn(img.shape, device=img.device)
    img_imag = std * randn(img.shape, device=img.device)
    return (img_real**2 + img_imag**2).sqrt()
