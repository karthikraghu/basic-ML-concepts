'''
Created on 20.06.2025

@author: Linda Schneider
'''

import numpy as np
import cv2

# do not import more modules!

def simpleAlignment(img, size=128):
    """
    param img: Input image (grayscale)
    param size: Size of the output canvas (default 128x128)
    return: Aligned image centered on a canvas of defined size
    """

    # Step 1: Resize input to standard canvas size
    img_resized = cv2.resize(img, (size, size))

    # Step 2: Apply Otsu's threshold
    _, thresh = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 3: Find coordinates of non-white (foreground) pixels
    coords = np.column_stack(np.where(thresh < 255))

    if coords.size == 0:
        # No foreground found; return black image
        return np.zeros((size, size), dtype=np.uint8)

    # Step 4: Get bounding box from coordinates
    top_left = coords.min(axis=0)
    bottom_right = coords.max(axis=0) + 1  # include bottom-right edge

    # Crop the region of interest from the original resized image
    roi = img_resized[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

    # Step 5: Resize ROI to half the canvas size
    target_dim = size // 2
    roi_resized = cv2.resize(roi, (target_dim, target_dim), interpolation=cv2.INTER_AREA)

    # Step 6: Center ROI in a new black canvas
    canvas = np.zeros((size, size), dtype=np.uint8)
    start_y = (size - target_dim) // 2
    start_x = (size - target_dim) // 2
    canvas[start_y:start_y + target_dim, start_x:start_x + target_dim] = roi_resized

    return canvas