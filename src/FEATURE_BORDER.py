import cv2
import numpy as np
from math import sqrt, floor, ceil, nan, pi
from skimage import color, exposure
from skimage.color import rgb2gray
from skimage.feature import blob_log
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops, perimeter
from skimage.transform import resize
from skimage.transform import rotate
from skimage import morphology
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage.color import rgb2hsv
from scipy.stats import circmean, circvar, circstd
from statistics import variance, stdev
from scipy.spatial import ConvexHull

def cut_mask(mask):
    '''Cut empty space from mask array such that it has smallest possible dimensions.

    Args:
        mask (numpy.ndarray): mask to cut

    Returns:
        cut_mask_ (numpy.ndarray): cut mask
    '''
    col_sums = np.sum(mask, axis=0)
    row_sums = np.sum(mask, axis=1)

    active_cols = [index for index, col_sum in enumerate(col_sums) if col_sum != 0]
    active_rows = [index for index, row_sum in enumerate(row_sums) if row_sum != 0]

    col_min, col_max = active_cols[0], active_cols[-1]
    row_min, row_max = active_rows[0], active_rows[-1]

    cut_mask_ = mask[row_min:row_max+1, col_min:col_max+1]

    return cut_mask_


def compactness_score(mask):
    # 1. Area is simple pixel count
    A = np.sum(mask > 0)
    
    if A == 0:
        return 0.0

    P = perimeter(mask, neighborhood=4)
    
    if P == 0:
        return 0.0

    # 3. Standard Polsby-Popper Formula
    compactness = (4 * np.pi * A) / (P**2)
    
    return min(float(compactness), 1.0)

def convexity_score(mask):


    '''Calculate convexity score between 0 and 1,
    with 0 indicating a smoother border and 1 a more crooked border.

    Args:
        image (numpy.ndarray): input masked image

    Returns:
        convexity_score (float): Float between 0 and 1 indicating convexity.
    '''

    if np.sum(mask) == 0:
        return None
    # Get coordinates of all pixels in the lesion mask
    coords = np.transpose(np.nonzero(mask))

    # Compute convex hull of lesion pixels
    hull = ConvexHull(coords)

    # Compute area of lesion mask
    lesion_area = np.count_nonzero(mask)

    # Compute area of convex hull
    convex_hull_area = hull.volume + hull.area

    # Compute convexity as ratio of lesion area to convex hull
    convexity = lesion_area / convex_hull_area

    return round(1-convexity, 3)

