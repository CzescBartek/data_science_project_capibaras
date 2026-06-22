import cv2
import numpy as np
def hair_coverage(img_gray):

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    whitehat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
    if np.sum(blackhat) > np.sum(whitehat):
        combined = blackhat
    else:
        combined = whitehat
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    total_area = img_gray.shape[0] * img_gray.shape[1]
    
    hair_area = cv2.countNonZero(hair_mask)
    
    coverage = hair_area / total_area
    
    return round(coverage, 4)

