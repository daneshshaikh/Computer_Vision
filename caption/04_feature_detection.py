"""
COMPUTER VISION - FEATURE DETECTION
====================================

Feature detection identifies distinctive points in images that can be used for:
- Image matching
- Object recognition
- Image stitching
- 3D reconstruction

Key Algorithms:
1. Harris Corner Detection - Detects corners
2. SIFT (Scale-Invariant Feature Transform) - Robust to scale and rotation
3. ORB (Oriented FAST and Rotated BRIEF) - Fast and efficient
4. FAST (Features from Accelerated Segment Test) - Very fast corner detection
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Example 1: Harris Corner Detection
def harris_corner_detection(img, block_size=2, ksize=3, k=0.04):
    """
    Detect corners using Harris corner detector
    
    Args:
        img: Input image (grayscale)
        block_size: Neighborhood size
        ksize: Aperture parameter for Sobel
        k: Harris detector free parameter
    
    Returns:
        Image with corners marked
    """
    # Detect corners
    corners = cv2.cornerHarris(img, block_size, ksize, k)
    
    # Dilate to mark corners
    corners = cv2.dilate(corners, None)
    
    # Create output image
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Mark corners in red
    img_color[corners > 0.01 * corners.max()] = [0, 0, 255]
    
    return img_color, corners

# Example 2: Shi-Tomasi Corner Detection (Good Features to Track)
def shi_tomasi_corners(img, max_corners=100, quality_level=0.01, min_distance=10):
    """
    Detect corners using Shi-Tomasi method
    
    Args:
        img: Input image (grayscale)
        max_corners: Maximum number of corners to detect
        quality_level: Quality threshold
        min_distance: Minimum distance between corners
    
    Returns:
        Image with corners marked
    """
    # Detect corners
    corners = cv2.goodFeaturesToTrack(img, max_corners, quality_level, min_distance)
    
    # Create output image
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Draw corners
    if corners is not None:
        corners = np.int0(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(img_color, (x, y), 5, (0, 255, 0), -1)
    
    return img_color, corners

# Example 3: SIFT Feature Detection
def sift_feature_detection(img):
    """
    Detect features using SIFT algorithm
    
    Args:
        img: Input image (grayscale)
    
    Returns:
        Image with keypoints, keypoints, descriptors
    """
    # Create SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(img, None)
    
    # Draw keypoints
    img_keypoints = cv2.drawKeypoints(img, keypoints, None, 
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return img_keypoints, keypoints, descriptors

# Example 4: ORB Feature Detection
def orb_feature_detection(img, n_features=500):
    """
    Detect features using ORB algorithm
    
    Args:
        img: Input image (grayscale)
        n_features: Number of features to detect
    
    Returns:
        Image with keypoints, keypoints, descriptors
    """
    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=n_features)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(img, None)
    
    # Draw keypoints
    img_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))
    
    return img_keypoints, keypoints, descriptors

# Example 5: FAST Feature Detection
def fast_feature_detection(img, threshold=10):
    """
    Detect features using FAST algorithm
    
    Args:
        img: Input image (grayscale)
        threshold: Threshold for corner detection
    
    Returns:
        Image with keypoints, keypoints
    """
    # Create FAST detector
    fast = cv2.FastFeatureDetector_create(threshold=threshold)
    
    # Detect keypoints
    keypoints = fast.detect(img, None)
    
    # Draw keypoints
    img_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(255, 0, 0))
    
    return img_keypoints, keypoints

# Example 6: Compare Feature Detectors
def compare_feature_detectors(img):
    """
    Compare different feature detection methods
    
    Args:
        img: Input image (grayscale)
    """
    # Apply different detectors
    harris_img, _ = harris_corner_detection(img)
    shi_tomasi_img, shi_corners = shi_tomasi_corners(img)
    sift_img, sift_kp, _ = sift_feature_detection(img)
    orb_img, orb_kp, _ = orb_feature_detection(img)
    fast_img, fast_kp = fast_feature_detection(img)
    
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(harris_img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Harris Corners')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cv2.cvtColor(shi_tomasi_img, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f'Shi-Tomasi ({len(shi_corners) if shi_corners is not None else 0} corners)')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(cv2.cvtColor(sift_img, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'SIFT ({len(sift_kp)} keypoints)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(orb_img, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'ORB ({len(orb_kp)} keypoints)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(cv2.cvtColor(fast_img, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(f'FAST ({len(fast_kp)} keypoints)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\nFeature Detection Statistics:")
    print("="*50)
    print(f"Shi-Tomasi corners: {len(shi_corners) if shi_corners is not None else 0}")
    print(f"SIFT keypoints: {len(sift_kp)}")
    print(f"ORB keypoints: {len(orb_kp)}")
    print(f"FAST keypoints: {len(fast_kp)}")

# Example 7: Feature Matching
def feature_matching_demo(img1, img2):
    """
    Demonstrate feature matching between two images
    
    Args:
        img1, img2: Input images (grayscale)
    """
    # Detect ORB features
    orb = cv2.ORB_create(nfeatures=1000)
    
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # Create BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(des1, des2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Draw matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    plt.figure(figsize=(15, 8))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(f'Feature Matching (Top 50 matches out of {len(matches)})')
    plt.axis('off')
    plt.show()
    
    return matches

# Demonstration
def demonstrate_feature_detection():
    """
    Demonstrate feature detection techniques
    """
    print("Demonstrating Feature Detection:")
    print("="*50)
    
    # Create a sample image with various features
    img = np.ones((400, 600), dtype=np.uint8) * 255
    
    # Draw shapes with corners and edges
    cv2.rectangle(img, (50, 50), (200, 200), 100, 3)
    cv2.rectangle(img, (100, 100), (150, 150), 80, -1)
    cv2.circle(img, (450, 150), 80, 120, 3)
    cv2.line(img, (300, 250), (500, 350), 90, 3)
    
    # Add some texture
    for i in range(20):
        x, y = np.random.randint(250, 550, 2)
        cv2.circle(img, (x, y), 5, 60, -1)
    
    print("\n1. Comparing feature detection methods...")
    compare_feature_detectors(img)
    
    print("\n2. Feature Detection Applications:")
    print("   - Image alignment and registration")
    print("   - Object tracking")
    print("   - Panorama stitching")
    print("   - 3D reconstruction")
    print("   - Augmented reality")
    print("   - Visual odometry")

if __name__ == "__main__":
    print(__doc__)
    demonstrate_feature_detection()
    
    print("\n✓ Feature detection complete!")
    print("Next: Learn about object detection in 05_object_detection.py")
