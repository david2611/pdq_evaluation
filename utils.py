import numpy as np


def generate_bounding_box_from_mask(mask):
    flat_x = np.any(mask, axis=0)
    flat_y = np.any(mask, axis=1)
    if not np.any(flat_x) and not np.any(flat_y):
        raise ValueError("No positive pixels found, cannot compute bounding box")
    xmin = np.argmax(flat_x)
    ymin = np.argmax(flat_y)
    xmax = len(flat_x) - 1 - np.argmax(flat_x[::-1])
    ymax = len(flat_y) - 1 - np.argmax(flat_y[::-1])
    return [xmin, ymin, xmax, ymax]