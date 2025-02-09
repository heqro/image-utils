import torch
from torch import Tensor
import torch.nn.functional as F

def roberts(image: Tensor):
    """
    Applies Roberts cross edge detection operator to an input image.

    Args:
        image (Tensor): Input image tensor of shape (B, C, H, W)

    Returns:
        tuple: Two tensors containing the diagonal edge detection results
        in two different directions (45° and 135°)
    """
    f1 = F.conv2d(
        input=image,
        weight=torch.tensor([[1.0, 0.0], [0.0, -1.0]], device=image.device).reshape(
            1, 1, 2, 2
        ),
    )
    f2 = F.conv2d(
        input=image,
        weight=torch.tensor([[0.0, 1.0], [-1.0, 0.0]], device=image.device).reshape(
            1, 1, 2, 2
        ),
    )
    return f1, f2


def prewitt(image: Tensor):
    """
    Applies Prewitt edge detection operator to an input image.

    Args:
        image (Tensor): Input image tensor of shape (B, C, H, W)

    Returns:
        tuple: Two tensors containing the horizontal and vertical edge detection results
    """
    x_filter = torch.tensor(
        [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], device=image.device
    ).reshape(1, 1, 3, 3)
    y_filter = torch.tensor(
        [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0], device=image.device
    ).reshape(1, 1, 3, 3)
    return F.conv2d(image, x_filter), F.conv2d(image, y_filter)


def sobel(image: Tensor):
    """
    Applies Sobel edge detection operator to an input image.

    Args:
        image (Tensor): Input image tensor of shape (B, C, H, W)

    Returns:
        tuple: Two tensors containing the horizontal and vertical edge detection results
    """
    x_filter = torch.tensor(
        [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0], device=image.device
    ).reshape(1, 1, 3, 3)
    y_filter = torch.tensor(
        [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0], device=image.device
    ).reshape(1, 1, 3, 3)
    conv_x = F.conv2d(input=image, weight=x_filter)
    conv_y = F.conv2d(input=image, weight=y_filter)
    return conv_x, conv_y


def kirsch(image: Tensor):
    """
    Applies Kirsch compass edge detection operator to an input image.

    Args:
        image (Tensor): Input image tensor of shape (B, C, H, W)

    Returns:
        list: Four tensors containing edge detection results in different directions
    """
    f1 = torch.tensor(
        [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], device=image.device
    ).reshape(1, 1, 3, 3)
    f2 = torch.tensor(
        [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0], device=image.device
    ).reshape(1, 1, 3, 3)
    f3 = torch.tensor(
        [-1.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 1.0], device=image.device
    ).reshape(1, 1, 3, 3)
    f4 = torch.tensor(
        [0.0, 1.0, 1.0, -1.0, 0.0, 1.0, -1.0, -1.0, 0.0], device=image.device
    ).reshape(1, 1, 3, 3)
    filters = [f1, f2, f3, f4]
    return [F.conv2d(input=image, weight=f) for f in filters]


def dct_3(image: Tensor):
    """
    Applies 3x3 Discrete Cosine Transform (DCT) filters to an input image.

    Args:
        image (Tensor): Input image tensor of shape (B, C, H, W)

    Returns:
        list: Eight tensors containing the DCT coefficients
    """
    filters = [
        torch.tensor(
            [-0.41, -0.41, -0.41, 0.0, 0.0, 0.0, 0.41, 0.41, 0.41], device=image.device
        ).reshape(1, 1, 3, 3),
        torch.tensor(
            [0.24, 0.24, 0.24, -0.47, -0.47, -0.47, 0.24, 0.24, 0.24],
            device=image.device,
        ).reshape(1, 1, 3, 3),
        torch.tensor(
            [0.41, 0.0, -0.41, 0.41, 0.0, -0.41, 0.41, 0.0, -0.41], device=image.device
        ).reshape(1, 1, 3, 3),
        torch.tensor(
            [-0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, -0.5], device=image.device
        ).reshape(1, 1, 3, 3),
        torch.tensor(
            [0.29, 0.0, -0.29, -0.58, 0.0, 0.58, 0.29, 0.0, -0.29], device=image.device
        ).reshape(1, 1, 3, 3),
        torch.tensor(
            [0.24, -0.47, 0.24, 0.24, -0.47, 0.24, 0.24, -0.47, 0.24],
            device=image.device,
        ).reshape(1, 1, 3, 3),
        torch.tensor(
            [-0.29, 0.58, -0.29, 0.0, 0.0, 0.0, 0.29, -0.58, 0.29], device=image.device
        ).reshape(1, 1, 3, 3),
        torch.tensor(
            [0.17, -0.33, 0.17, -0.33, 0.67, -0.33, 0.17, -0.33, 0.17],
            device=image.device,
        ).reshape(1, 1, 3, 3),
    ]
    return [F.conv2d(input=image, weight=f) for f in filters]


def grads(image: Tensor):
    """
    Computes simple gradient operators in x and y directions.

    Args:
        image (Tensor): Input image tensor of shape (B, C, H, W) or (C, H, W)

    Returns:
        tuple: Two tensors containing the x and y gradients with same padding
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
        print(
            "grads - WARN: input image has 3 dimensions instead of 4. Adding new dim."
        )
    x_filter = torch.tensor(
        [0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0], device=image.device
    ).reshape(1, 1, 3, 3)
    y_filter = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0], device=image.device
    ).reshape(1, 1, 3, 3)
    dx = F.conv2d(image, x_filter, padding="same")
    dy = F.conv2d(image, y_filter, padding="same")
    return dx, dy


def derivative5(image: Tensor):
    """
    Applies 5x5 derivative filters for precise gradient computation.

    Args:
        image (Tensor): Input image tensor of shape (B, C, H, W)

    Returns:
        tuple: Two tensors containing the x and y derivatives with same padding
    """
    k_1 = torch.tensor([0.037659, 0.249153, 0.426375, 0.249153, 0.037659])
    k_2 = torch.tensor(
        [
            0.109604,
            0.276691,
            0.0,
            -0.276691,
            -0.109604,
        ]
    )
    dx = F.conv2d(
        F.conv2d(image, k_1.view(1, 1, -1, 1), padding="same"),
        k_2.view(1, 1, 1, -1),
        padding="same",
    )
    dy = F.conv2d(
        F.conv2d(image, k_2.view(1, 1, -1, 1), padding="same"),
        k_1.view(1, 1, 1, -1),
        padding="same",
    )
    return dx, dy


def derivative7(image: Tensor):
    """
    Applies 7x7 derivative filters for precise gradient computation.

    Args:
        image (Tensor): Input image tensor of shape (B, C, H, W)

    Returns:
        tuple: Two tensors containing the x and y derivatives with same padding
    """
    k_1 = torch.tensor(
        [0.004711, 0.069321, 0.245410, 0.361117, 0.245410, 0.069321, 0.004711]
    )
    k_2 = torch.tensor(
        [0.018708, 0.125376, 0.193091, 0.0, -0.193091, -0.125376, -0.018708]
    )
    dx = F.conv2d(
        F.conv2d(image, k_1.view(1, 1, -1, 1), padding="same"),
        k_2.view(1, 1, 1, -1),
        padding="same",
    )
    dy = F.conv2d(
        F.conv2d(image, k_2.view(1, 1, -1, 1), padding="same"),
        k_1.view(1, 1, 1, -1),
        padding="same",
    )
    return dx, dy


def anscombe_transform(x: torch.Tensor, sigma: float):
    """
    Applies the Anscombe transform to stabilize variance in noisy image data.

    Args:
        x (torch.Tensor): Input tensor
        sigma (float): Noise standard deviation

    Returns:
        torch.Tensor: Transformed tensor with stabilized variance
    """
    return torch.sqrt(torch.clip(x**2 - sigma**2, 0))
