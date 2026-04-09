import torch


def to_kspace(img: torch.Tensor) -> torch.Tensor:
    """
    Slingshots a magnitude image `img` to k-space.
    """
    return torch.fft.fftshift(torch.fft.fft2(img))


def from_kspace(kspace_img: torch.Tensor) -> torch.Tensor:
    """
    Sends a k-space image `kspace_img` to the complex plane.
    """
    # Undo the centering shift
    undo_shift = torch.fft.ifftshift(kspace_img)
    # Specify the last two dimensions to handle potential (C, H, W) shapes
    return torch.fft.ifft2(undo_shift, dim=(-2, -1))


def to_magnitude(complex_from_kspace: torch.Tensor) -> torch.Tensor:
    """
    Sends an image from the complex plane, `complex_from_kspace`, to magnitude image.
    """
    return torch.abs(complex_from_kspace)


def calculate_radial_profile(
    k_space: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    H, W = k_space.shape[-2], k_space.shape[-1]
    device = k_space.device

    Y, X = torch.meshgrid(
        torch.arange(H, device=device) - H // 2,
        torch.arange(W, device=device) - W // 2,
        indexing="ij",
    )

    r = torch.sqrt(X**2 + Y**2).round().long()
    magnitude = torch.abs(k_space)

    if mask is not None:
        # Boolean indexing to extract only valid frequencies
        valid_idx = mask.bool()
        r = r[valid_idx]
        magnitude = magnitude[valid_idx]

        if r.numel() == 0:
            return torch.tensor([], device=device)

    radial_sum = torch.bincount(r.flatten(), weights=magnitude.flatten())
    radial_count = torch.bincount(r.flatten())

    return radial_sum / radial_count.clamp(min=1)


def estimate_and_project_decay(
    measured_profile: torch.Tensor, full_length: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    def cosine_taper(taper_length: int, min_val: float):
        t = torch.linspace(0, 1, taper_length)
        return min_val + (1.0 - min_val) * (0.5 * (1.0 + torch.cos(torch.pi * t)))

    def poisson_taper(taper_length: int, decay_rate: float):
        t = torch.linspace(0, 1, taper_length)
        return torch.exp(-decay_rate * t)

    """
    Estimates power law decay from the measured profile and projects it to full_length.
    """
    measured_len = len(measured_profile)
    start = measured_len - 20
    r_measured = torch.arange(
        start, measured_len, device=measured_profile.device, dtype=torch.float32
    )
    E_measured = measured_profile[start:]

    # Transform to log-log space
    x = torch.log(r_measured)
    y = torch.log(E_measured + 1e-8)

    # Linear Regression
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)

    covariance = torch.sum((x - x_mean) * (y - y_mean))
    variance = torch.sum((x - x_mean) ** 2)

    slope = covariance / variance
    intercept = y_mean - slope * x_mean

    # Project across full spectrum
    r_all = torch.arange(
        measured_len, full_length, device=measured_profile.device, dtype=torch.float32
    )
    target_log_E = slope * torch.log(r_all) + intercept
    target_decay = torch.exp(target_log_E)

    # Tapering of prediction to better fit power law
    taper_length = full_length - measured_len
    # target_decay = target_decay * cosine_taper(taper_length, 0.15)
    target_decay = target_decay * poisson_taper(taper_length, 1.2).to(
        target_decay.device
    )

    # Pad DC component to align indices with the original profile
    projected_profile = torch.zeros(full_length, device=measured_profile.device)
    projected_profile[:measured_len] = measured_profile
    projected_profile[measured_len:] = target_decay

    return projected_profile, target_decay, measured_len
