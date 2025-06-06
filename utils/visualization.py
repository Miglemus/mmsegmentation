from PIL import Image
import numpy as np
import matplotlib.pyplot as plt  # Add this import

def show_image(image_path: str, mask_path: str):
    if isinstance(image_path, str) and isinstance(mask_path, str):
        image_path = [image_path]
        mask_path = [mask_path]

    elif isinstance(image_path, list) and isinstance(mask_path, list):
        pass

    else: raise ValueError("different image and mask types")

    for image, mask in zip(image_path, mask_path):
        # Load the image
        img = Image.open(image).convert("RGB")
        img = np.array(img)

        # Load the mask
        msk = Image.open(mask).convert("L")
        msk = np.array(msk)

        plt.imshow(img)
        # Overlay: red color with alpha=0.5 where mask > 0
        plt.imshow(
            np.dstack([
                np.ones_like(msk) * 255,  # Red channel
                np.zeros_like(msk),       # Green channel
                np.zeros_like(msk),       # Blue channel
                (msk > 0) * 128           # Alpha channel (0 or 128)
            ]),
            alpha=0.5
        )
        plt.axis('off')
        plt.show()


def apply_mask(image, mask):
    """
    Apply a mask to an image.
    
    Args:
        image (np.ndarray): The input image.
        mask (np.ndarray): The mask to apply.
        
    Returns:
        np.ndarray: The masked image.
    """
    if len(image.shape) == 2:  # Grayscale image
        masked_image = np.where(mask > 0, image, 0)
    else:  # Color image
        masked_image = np.where(mask[..., None] > 0, image, 0)
    
    return masked_image
