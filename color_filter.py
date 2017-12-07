import cv2
import numpy as np


def red_filtered(image):
    lbound = np.array([0, 50, 50], dtype = np.uint8)
    ubound = np.array([10, 255, 255], dtype = np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lbound, ubound)
    rgb_img = np.zeros((mask.shape[0], mask.shape[1], 3), 'uint8')
    rgb_img[..., 0] = mask
    rgb_img[..., 1] = mask
    rgb_img[..., 2] = mask
    return mask, rgb_img
    



