from torch import Tensor, randn, Generator, empty_like


def add_gaussian_noise(
    img: Tensor, std: float, avg: float = 0, seed: int | None = None
) -> Tensor:
    """Add Gaussian noise to a tensor.

    Args:
        img (Tensor): Image tensor to add noise to.
        std (float): Standard deviation of the noise.
        avg (float): Mean of the noise (default 0).
        seed (int | None): Optional seed for reproducible noise. Defaults to None.

    Returns:
        Tensor: Image tensor with added Gaussian noise.
    """
    gen = Generator(device=img.device)
    if seed is not None:
        gen.manual_seed(seed)
    noise = avg + std * randn(img.shape, generator=gen, device=img.device)
    return img + noise


def add_rician_noise(img: Tensor, std: float, seed: int | None = None) -> Tensor:
    """Add Rician noise to a tensor, optionally seeded.

    Args:
        img (Tensor): Image tensor to add noise to.
        std (float): Standard deviation of the noise.
        seed (int | None): Optional seed for reproducible noise. Defaults to None.

    Returns:
        Tensor: Image tensor with added Rician noise.
    """
    if seed is not None:
        gen = Generator(device=img.device)
        gen.manual_seed(seed)
    else:
        gen = None
    img_real = img + empty_like(img).normal_(generator=gen, std=std)
    img_imag = empty_like(img).normal_(generator=gen, std=std)
    return (img_real**2 + img_imag**2).sqrt()
