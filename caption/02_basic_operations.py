"""
COMPUTER VISION - BASIC IMAGE OPERATIONS
=========================================

Learn fundamental image manipulation techniques:
- Resizing and cropping
- Rotating and flipping
- Color space conversions
- Image filtering
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Example 1: Resizing Images
def resize_image(img, width=None, height=None, scale=None):
    """
    Resize an image using different methods
    
    Args:
        img: Input image
        width: Target width (optional)
        height: Target height (optional)
        scale: Scale factor (optional)
    """
    if scale is not None:
        # Resize by scale factor
        new_width = int(img.shape[1] * scale)
        new_height = int(img.shape[0] * scale)
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    elif width is not None and height is not None:
        # Resize to specific dimensions
        resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    else:
        raise ValueError("Provide either scale or width and height")
    
    return resized

# Example 2: Cropping Images
def crop_image(img, x, y, width, height):
    """
    Crop a rectangular region from an image
    
    Args:
        img: Input image
        x, y: Top-left corner coordinates
        width, height: Crop dimensions
    """
    cropped = img[y:y+height, x:x+width]
    return cropped

# Example 3: Rotating Images
def rotate_image(img, angle):
    """
    Rotate an image by a given angle
    
    Args:
        img: Input image
        angle: Rotation angle in degrees (positive = counter-clockwise)
    """
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform rotation
    rotated = cv2.warpAffine(img, rotation_matrix, (width, height))
    
    return rotated

# Example 4: Flipping Images
def flip_image(img, flip_code):
    """
    Flip an image
    
    Args:
        img: Input image
        flip_code: 0 = vertical, 1 = horizontal, -1 = both
    """
    flipped = cv2.flip(img, flip_code)
    return flipped

# Example 5: Color Space Conversions
def convert_color_spaces(img):
    """
    Convert image to different color spaces
    
    Args:
        img: Input image in BGR format
    """
    # BGR to RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # BGR to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # BGR to HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('RGB')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(hsv)
    axes[1, 0].set_title('HSV')
    axes[1, 0].axis('off')
    
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return rgb, gray, hsv

# Example 6: Image Filtering (Blurring)
def apply_filters(img):
    """
    Apply various filters to an image
    
    Args:
        img: Input image
    """
    # Gaussian Blur
    gaussian = cv2.GaussianBlur(img, (15, 15), 0)
    
    # Median Blur
    median = cv2.medianBlur(img, 15)
    
    # Bilateral Filter (preserves edges)
    bilateral = cv2.bilateralFilter(img, 15, 75, 75)
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Gaussian Blur')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Median Blur')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Bilateral Filter')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Example 7: Brightness and Contrast Adjustment
def adjust_brightness_contrast(img, brightness=0, contrast=0):
    """
    Adjust brightness and contrast of an image
    
    Args:
        img: Input image
        brightness: Brightness adjustment (-100 to 100)
        contrast: Contrast adjustment (-100 to 100)
    """
    # Convert to float for calculations
    adjusted = img.astype(np.float32)
    
    # Apply brightness
    adjusted = adjusted + brightness
    
    # Apply contrast
    adjusted = adjusted * (1 + contrast / 100.0)
    
    # Clip values to valid range
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    
    return adjusted

# Demonstration
def demonstrate_operations():
    """
    Demonstrate all basic operations
    """
    # Create a sample image
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Draw some shapes
    cv2.rectangle(img, (50, 50), (250, 250), (255, 0, 0), -1)
    cv2.circle(img, (450, 150), 100, (0, 255, 0), -1)
    cv2.line(img, (50, 300), (550, 350), (0, 0, 255), 5)
    
    print("Demonstrating Basic Operations:")
    print("="*50)
    
    # 1. Resize
    print("\n1. Resizing...")
    resized = resize_image(img, scale=0.5)
    print(f"   Original size: {img.shape[:2]}")
    print(f"   Resized size: {resized.shape[:2]}")
    
    # 2. Crop
    print("\n2. Cropping...")
    cropped = crop_image(img, 100, 100, 200, 200)
    print(f"   Cropped region: 200x200 pixels")
    
    # 3. Rotate
    print("\n3. Rotating...")
    rotated = rotate_image(img, 45)
    print(f"   Rotated by 45 degrees")
    
    # 4. Flip
    print("\n4. Flipping...")
    flipped = flip_image(img, 1)
    print(f"   Flipped horizontally")
    
    # 5. Brightness/Contrast
    print("\n5. Adjusting brightness and contrast...")
    adjusted = adjust_brightness_contrast(img, brightness=30, contrast=20)
    print(f"   Brightness: +30, Contrast: +20")
    
    # Display all results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Resized (50%)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Cropped')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Rotated (45°)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Flipped Horizontally')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Brightness/Contrast Adjusted')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print(__doc__)
    demonstrate_operations()
    
    print("\n✓ Basic operations complete!")
    print("Next: Learn about edge detection in 03_edge_detection.py")
