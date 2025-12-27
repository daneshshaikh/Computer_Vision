"""
COMPUTER VISION - INTRODUCTION
===============================

Computer Vision is a field of artificial intelligence that enables computers
to interpret and understand visual information from the world.

Key Concepts:
1. Image Processing - Manipulating and analyzing digital images
2. Feature Detection - Identifying important patterns in images
3. Object Recognition - Identifying objects in images
4. Image Segmentation - Dividing images into meaningful regions
5. Deep Learning - Using neural networks for vision tasks

This learning path will cover:
- Basic image operations
- Classical computer vision techniques
- Deep learning for computer vision
- Practical applications
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Example 1: Loading and Displaying an Image
def load_and_display_image(image_path):
    """
    Load an image using OpenCV and display it
    
    Args:
        image_path: Path to the image file
    """
    # Load image
    img = cv2.imread(image_path)
    
    # Convert BGR to RGB (OpenCV loads in BGR format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Display image
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    plt.show()
    
    return img

# Example 2: Creating a Simple Image
def create_sample_image():
    """
    Create a simple colored image using NumPy
    """
    # Create a 300x300 RGB image
    height, width = 300, 300
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create gradient effect
    for i in range(height):
        for j in range(width):
            img[i, j] = [i % 256, j % 256, (i + j) % 256]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title('Generated Gradient Image')
    plt.axis('off')
    plt.show()
    
    return img

# Example 3: Basic Image Properties
def analyze_image_properties(img):
    """
    Analyze and print basic image properties
    
    Args:
        img: Input image (NumPy array)
    """
    print("Image Properties:")
    print(f"Shape: {img.shape}")
    print(f"Height: {img.shape[0]} pixels")
    print(f"Width: {img.shape[1]} pixels")
    print(f"Channels: {img.shape[2] if len(img.shape) > 2 else 1}")
    print(f"Data type: {img.dtype}")
    print(f"Min pixel value: {img.min()}")
    print(f"Max pixel value: {img.max()}")
    print(f"Mean pixel value: {img.mean():.2f}")

if __name__ == "__main__":
    print(__doc__)
    
    # Create and analyze a sample image
    print("\n" + "="*50)
    print("Creating a sample image...")
    print("="*50)
    
    sample_img = create_sample_image()
    analyze_image_properties(sample_img)
    
    print("\n✓ Introduction complete!")
    print("Next: Learn about basic image operations in 02_basic_operations.py")
