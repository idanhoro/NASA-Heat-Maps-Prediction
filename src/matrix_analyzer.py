import os
import sys
from io import BytesIO

import numpy as np
from PIL import Image
from scipy.spatial import distance_matrix

from src.crawler import image_url_to_mat, get_by_url


def map_analyzer(map_url: str, clean_img_arr: np.ndarray, scale_arr: np.ndarray) -> np.ndarray:
    img_arr: np.ndarray = image_url_to_mat(map_url)
    difference_array: np.ndarray = np.sqrt(np.sum((img_arr - clean_img_arr) ** 2, axis=2))
    mask_array: np.ndarray = difference_array != 0
    img_arr_subset: np.ndarray = img_arr[mask_array]

    dist_matrix: np.ndarray = np.c_[difference_array[mask_array], distance_matrix(img_arr_subset, scale_arr)]

    min_dist_indices: np.ndarray = (dist_matrix.argmin(axis=-1) - 1)  # Shift the values (scale indices) by -1
    scale_pixels_final: np.ndarray = np.full(shape=(350, 700), fill_value=-1)  # (350,700) index matrix
    scale_pixels_final[mask_array] = min_dist_indices
    scale_pixels_final_reshaped: np.ndarray = scale_pixels_final.reshape(-1)

    return scale_pixels_final_reshaped


def color_mapper(scale_url: str) -> np.ndarray:
    print("Scale URL: ", scale_url)

    scale_img = Image.open(BytesIO(get_by_url(scale_url).content)).convert("RGB")
    color_strip = np.asarray(scale_img)[int(scale_img.size[1] / 2) - 1]  # Take only the middle strip -- height / 2

    (start, end) = (scale_searcher(color_strip), scale_searcher(color_strip[::-1]))

    if start == -1 or end == -1:
        sys.exit(1)

    rgb_scale = color_strip[start:-end]

    return rgb_scale


def scale_searcher(scale_arr: np.ndarray) -> int:
    ret = 0

    for idx, rgb in enumerate(scale_arr):
        ret = -1

        if any(v != 255 for v in rgb):
            return idx + 1

    return ret


def create_outliers_mask(categories_by_month_dict) -> np.ndarray:
    """
    Creates a `outliers_mask.npy` which contains a mask of all the land pixels.
    Ran once on the "Vegetation" and "Land Surface Temperature" maps
    in order to create a mask which contains only the "valuable" pixels.

    :param categories_by_month_dict:
     A dictionary of categories of which each contains a dictionary indexed by tuple[year,month]
     its' values contain arrays of all scale indexes.
    :return: A mask array containing `True` if all values in an index != -1 and False otherwise.
    """

    zeros = np.zeros(245000, dtype=np.int32)
    ones = np.ones(245000, dtype=np.int32)

    for cat in categories_by_month_dict:
        for month in categories_by_month_dict[cat]:
            zeros = zeros + categories_by_month_dict[cat][month] + ones

    # If 0 + 1 + x == 0 => x == -1 --> x is an outlier.
    outliers_mask = (zeros != 0).reshape(350, 700)
    np.save(os.path.join("assets", "outliers_mask.npy"), outliers_mask)

    return outliers_mask
