"""
COMPUTER VISION - EDGE DETECTION
=================================

Edge detection is a fundamental technique in computer vision used to identify
boundaries of objects within images.

Key Algorithms:
1. Canny Edge Detection - Multi-stage algorithm for optimal edge detection
2. Sobel Operator - Gradient-based edge detection
3. Laplacian - Second derivative-based detection
4. Prewitt Operator - Similar to Sobel
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Example 1: Canny Edge Detection
def canny_edge_detection(img, low_threshold=50, high_threshold=150):
    """
    Apply Canny edge detection algorithm
    
    Args:
        img: Input image (grayscale)
        low_threshold: Lower threshold for edge detection
        high_threshold: Upper threshold for edge detection
    
    Returns:
        Edge-detected image
    """
    edges = cv2.Canny(img, low_threshold, high_threshold)
    return edges

# Example 2: Sobel Edge Detection
def sobel_edge_detection(img):
    """
    Apply Sobel edge detection
    
    Args:
        img: Input image (grayscale)
    
    Returns:
        Edge-detected images (x, y, and combined)
    """
    # Sobel in X direction
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = np.uint8(np.absolute(sobelx))
    
    # Sobel in Y direction
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobely = np.uint8(np.absolute(sobely))
    
    # Combined Sobel
    sobel_combined = cv2.bitwise_or(sobelx, sobely)
    
    return sobelx, sobely, sobel_combined

# Example 3: Laplacian Edge Detection
def laplacian_edge_detection(img):
    """
    Apply Laplacian edge detection
    
    Args:
        img: Input image (grayscale)
    
    Returns:
        Edge-detected image
    """
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    return laplacian

# Example 4: Compare All Edge Detection Methods
def compare_edge_detection_methods(img):
    """
    Compare different edge detection methods
    
    Args:
        img: Input image (BGR)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply different edge detection methods
    canny = canny_edge_detection(blurred)
    sobelx, sobely, sobel_combined = sobel_edge_detection(blurred)
    laplacian = laplacian_edge_detection(blurred)
    
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original (Grayscale)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(canny, cmap='gray')
    axes[0, 1].set_title('Canny Edge Detection')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(sobelx, cmap='gray')
    axes[0, 2].set_title('Sobel X')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(sobely, cmap='gray')
    axes[1, 0].set_title('Sobel Y')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(sobel_combined, cmap='gray')
    axes[1, 1].set_title('Sobel Combined')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(laplacian, cmap='gray')
    axes[1, 2].set_title('Laplacian')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Example 5: Adaptive Canny Edge Detection
def adaptive_canny(img, sigma=0.33):
    """
    Apply Canny edge detection with automatic threshold calculation
    
    Args:
        img: Input image (grayscale)
        sigma: Sigma value for threshold calculation
    
    Returns:
        Edge-detected image
    """
    # Calculate median of pixel intensities
    median = np.median(img)
    
    # Calculate thresholds
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    
    # Apply Canny
    edges = cv2.Canny(img, lower, upper)
    
    return edges

# Example 6: Edge Detection with Different Parameters
def explore_canny_parameters(img):
    """
    Explore how different parameters affect Canny edge detection
    
    Args:
        img: Input image (BGR)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Different threshold combinations
    params = [
        (30, 100, 'Low Thresholds'),
        (50, 150, 'Medium Thresholds'),
        (100, 200, 'High Thresholds'),
        (150, 250, 'Very High Thresholds')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, (low, high, title) in enumerate(params):
        edges = canny_edge_detection(blurred, low, high)
        axes[idx].imshow(edges, cmap='gray')
        axes[idx].set_title(f'{title}\n(Low={low}, High={high})')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

# Demonstration
def demonstrate_edge_detection():
    """
    Demonstrate edge detection techniques
    """
    print("Demonstrating Edge Detection:")
    print("="*50)
    
    # Create a sample image with shapes
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Draw shapes
    cv2.rectangle(img, (50, 50), (250, 250), (100, 100, 100), -1)
    cv2.circle(img, (450, 150), 80, (150, 150, 150), -1)
    cv2.ellipse(img, (300, 300), (100, 50), 45, 0, 360, (120, 120, 120), -1)
    
    # Add some noise
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    
    print("\n1. Comparing edge detection methods...")
    compare_edge_detection_methods(img)
    
    print("\n2. Exploring Canny parameters...")
    explore_canny_parameters(img)
    
    print("\n3. Edge Detection Applications:")
    print("   - Object boundary detection")
    print("   - Feature extraction")
    print("   - Image segmentation")
    print("   - Shape analysis")
    print("   - Pattern recognition")

if __name__ == "__main__":
    print(__doc__)
    demonstrate_edge_detection()
    
    print("\n✓ Edge detection complete!")
    print("Next: Learn about feature detection in 04_feature_detection.py")
