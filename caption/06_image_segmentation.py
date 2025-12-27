"""
COMPUTER VISION - IMAGE SEGMENTATION
=====================================

Image segmentation divides an image into meaningful regions.

Techniques: Thresholding, Watershed, GrabCut, K-Means, Mean Shift

Applications: Medical imaging, autonomous driving, background removal
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def kmeans_segmentation(img, k=3):
    """Segment image using K-Means clustering"""
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, 
                                    cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(img.shape)
    
    return segmented_image

def thresholding_demo(img_gray):
    """Apply various thresholding techniques"""
    _, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    _, otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    
    return binary, otsu, adaptive

if __name__ == "__main__":
    print(__doc__)
    print("\n✓ Image segmentation module ready!")
    print("Next: 07_deep_learning_intro.py")
