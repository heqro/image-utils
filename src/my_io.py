import torchvision
from torch import Tensor, load, float32


def __load_image__(path: str, mode: torchvision.io.ImageReadMode) -> Tensor:
    """
    Internal function used by the load_gray_image and load_rgb_image functions.
    Loads an image from a file path and normalizes it to [0,1] range.

    Args:
        path (str): Path to the image file
        mode (torchvision.io.ImageReadMode): Mode to read the image in (GRAY or RGB)

    Returns:
        torch.Tensor: Normalized image tensor with values in [0,1]
    """
    return torchvision.io.read_image(path, mode) / 255


def load_rgb_image(path: str) -> Tensor:
    """
    Load an RGB image from a file path.

    Args:
        path (str): Path to the image file

    Returns:
        torch.Tensor: Normalized RGB image tensor with values in [0,1]
    """
    return __load_image__(path, torchvision.io.image.ImageReadMode.RGB)


def load_gray_image(path: str, as_mask=False) -> Tensor:
    """
    Load a grayscale image from a file path and optionally processes it as a mask.

    Args:
        path (str): Path to the image file
        is_mask (bool): If True, binarizes the image by setting all non-zero values to the maximum value

    Returns:
        torch.Tensor: Normalized grayscale image tensor with values in [0,1]
    """
    img = __load_image__(path, torchvision.io.image.ImageReadMode.GRAY)
    if as_mask:
        img[img > 0] = img.max()
    return img


def load_serialized_image(path: str, is_mask=False, normalize=True) -> Tensor:
    """
    Load a serialized image tensor from a file path and optionally processes it as a mask and/or normalizes it.

    Args:
        path (str): Path to the serialized tensor file
        is_mask (bool): If True, binarizes the image by setting all non-zero values to the maximum value
        normalize (bool): If True, normalizes the image values to [0,1] range

    Returns:
        torch.Tensor: Processed image tensor
    """
    img = load(path, weights_only=False).to(dtype=float32)
    if is_mask:
        img[img > 0] = img.max()
    if normalize:
        img = (img - img.min()) / (img.max() - img.min())
    return img


def print_image(img: ndarray, file_name: str | None):
    if file_name is None:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        plt.close()
    else:
        from imageio import imwrite

        imwrite(uri=file_name, im=img)
