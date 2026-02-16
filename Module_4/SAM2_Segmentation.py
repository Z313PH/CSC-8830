import cv2
import numpy as np

def iou_dice(mask_a_path, mask_b_path):
    A = cv2.imread(mask_a_path, cv2.IMREAD_GRAYSCALE) > 0
    B = cv2.imread(mask_b_path, cv2.IMREAD_GRAYSCALE) > 0

    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    iou = inter / (union + 1e-12)

    dice = (2 * inter) / (A.sum() + B.sum() + 1e-12)
    return iou, dice
