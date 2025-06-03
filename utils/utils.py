import cv2
import os
import numpy as np
# import mmcv
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import Image

def coco_to_mask(coco_path, output_dir, image_dir, class_map):
    os.makedirs(output_dir, exist_ok=True)

    coco = COCO(coco_path)

    for img in coco.imgs.values():
        img_id = img['id']
        img_filename = img['file_name']
        height = img['height']
        width = img['width']

        mask = np.zeros((height, width), dtype=np.uint8)

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            category_id = ann['category_id']
            class_id = class_map[category_id]

            if isinstance(ann['segmentation'], list):
                rles = maskUtils.frPyObjects(ann['segmentation'], height, width)
                rle = maskUtils.merge(rles)
            elif isinstance(ann['segmentation']['counts'], list):
                rle = maskUtils.frPyObjects([ann['segmentation']], height, width)
            else:
                rle = ann['segmentation']

            m = maskUtils.decode(rle)
            mask[m == 1] = class_id

        out_path = os.path.join(output_dir, os.path.splitext(img_filename)[0] + '.png')
        Image.fromarray(mask).save(out_path)

    print(f"Done. Masks saved to: {output_dir}")


def polygon_to_mask(segmentation, image_height, image_width):
    """
    Convert polygon segmentation to a binary mask.
    
    Parameters:
        segmentation (list): List of polygons (each polygon is a list of [x, y] points).
        image_height (int): Height of the image.
        image_width (int): Width of the image.

    Returns:
        np.ndarray: Binary mask of shape (image_height, image_width).
    """
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    
    for polygon in segmentation:
        # Convert polygon to numpy array of shape (num_points, 1, 2)
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], color=1)

    return mask

def mask_to_polygon(mask, epsilon=1.0):
    """
    Convert a binary mask to COCO-style polygon segmentation.

    Parameters:
        mask (np.ndarray): Binary mask (2D numpy array).
        epsilon (float): Approximation accuracy. Smaller = more detailed polygon.

    Returns:
        List[List[float]]: List of polygon(s), each polygon is a flat list of coordinates.
    """
    # Ensure mask is uint8 and binary
    mask = (mask > 0).astype(np.uint8)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if contour.shape[0] >= 3:  # at least 3 points to form a polygon
            approx = cv2.approxPolyDP(contour, epsilon, True)
            polygon = approx.reshape(-1).tolist()
            polygons.append(polygon)
    
    return polygons


def train_val_test_split(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split data into train, validation, and test sets.

    Parameters:
        data (list): List of data items to be split.
        train_ratio (float): Proportion of data to use for training.
        val_ratio (float): Proportion of data to use for validation.
        test_ratio (float): Proportion of data to use for testing.

    Returns:
        tuple: Three lists containing the train, validation, and test sets.
    """
    assert all(ratio >= 0 for ratio in [train_ratio, val_ratio, test_ratio]), "Ratios must be non-negative."
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1."

    np.random.shuffle(data)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data
