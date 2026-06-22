import cv2
import numpy as np

def remove_penmarks(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([80, 10, 5])    
    upper_blue = np.array([180, 255, 255])

    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    img_out = cv2.inpaint(img, mask, 1, cv2.INPAINT_TELEA)

    return img_out