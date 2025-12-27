"""
COMPUTER VISION - OBJECT DETECTION
===================================

Object detection identifies and locates objects within images.

Techniques Covered:
1. Template Matching - Simple pattern matching
2. Haar Cascade Classifiers - Classical face/object detection
3. Contour Detection - Shape-based detection
4. Color-based Detection - Detecting objects by color

Modern Deep Learning Approaches (Overview):
- YOLO (You Only Look Once)
- SSD (Single Shot Detector)
- Faster R-CNN
- RetinaNet
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Example 1: Template Matching
def template_matching(img, template):
    """
    Find template in image using template matching
    
    Args:
        img: Input image (grayscale)
        template: Template to find
    
    Returns:
        Image with detected template marked
    """
    # Get template dimensions
    h, w = template.shape
    
    # Apply template matching
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    
    # Set threshold
    threshold = 0.8
    locations = np.where(result >= threshold)
    
    # Create output image
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Draw rectangles around matches
    for pt in zip(*locations[::-1]):
        cv2.rectangle(img_color, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
    
    return img_color, result

# Example 2: Contour Detection
def contour_detection(img):
    """
    Detect and analyze contours in an image
    
    Args:
        img: Input image (grayscale)
    
    Returns:
        Image with contours, contours list
    """
    # Apply threshold
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create output image
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Draw all contours
    cv2.drawContours(img_color, contours, -1, (0, 255, 0), 2)
    
    return img_color, contours

# Example 3: Shape Detection using Contours
def detect_shapes(img):
    """
    Detect and classify shapes in an image
    
    Args:
        img: Input image (grayscale)
    
    Returns:
        Image with labeled shapes
    """
    # Apply threshold
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create output image
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    for contour in contours:
        # Calculate area
        area = cv2.contourArea(contour)
        
        if area > 100:  # Filter small contours
            # Approximate contour
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(approx)
            
            # Classify shape based on number of vertices
            vertices = len(approx)
            shape = "Unknown"
            
            if vertices == 3:
                shape = "Triangle"
            elif vertices == 4:
                aspect_ratio = float(w) / h
                shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
            elif vertices > 4:
                shape = "Circle"
            
            # Draw contour and label
            cv2.drawContours(img_color, [contour], -1, (0, 255, 0), 2)
            cv2.putText(img_color, shape, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return img_color

# Example 4: Color-based Object Detection
def detect_by_color(img, lower_color, upper_color):
    """
    Detect objects based on color range
    
    Args:
        img: Input image (BGR)
        lower_color: Lower bound of color (HSV)
        upper_color: Upper bound of color (HSV)
    
    Returns:
        Mask and detected objects
    """
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create mask
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Apply morphological operations to clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes
    img_result = img.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Filter small areas
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return mask, img_result

# Example 5: Circle Detection using Hough Transform
def detect_circles(img):
    """
    Detect circles using Hough Circle Transform
    
    Args:
        img: Input image (grayscale)
    
    Returns:
        Image with detected circles
    """
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (9, 9), 2)
    
    # Detect circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=100, param2=30, minRadius=10, maxRadius=100)
    
    # Create output image
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Draw circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            # Draw circle
            cv2.circle(img_color, center, radius, (0, 255, 0), 2)
            # Draw center
            cv2.circle(img_color, center, 2, (0, 0, 255), 3)
    
    return img_color, circles

# Example 6: Blob Detection
def blob_detection(img):
    """
    Detect blobs in an image
    
    Args:
        img: Input image (grayscale)
    
    Returns:
        Image with detected blobs
    """
    # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    
    # Filter by Area
    params.filterByArea = True
    params.minArea = 100
    
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5
    
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.5
    
    # Create detector
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs
    keypoints = detector.detect(img)
    
    # Draw detected blobs
    img_with_blobs = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return img_with_blobs, keypoints

# Demonstration
def demonstrate_object_detection():
    """
    Demonstrate object detection techniques
    """
    print("Demonstrating Object Detection:")
    print("="*50)
    
    # Create a sample image with various shapes
    img = np.ones((500, 700), dtype=np.uint8) * 255
    
    # Draw different shapes
    cv2.rectangle(img, (50, 50), (150, 150), 0, -1)  # Square
    cv2.rectangle(img, (200, 50), (350, 120), 0, -1)  # Rectangle
    cv2.circle(img, (450, 100), 50, 0, -1)  # Circle
    
    # Triangle
    pts = np.array([[100, 250], [50, 350], [150, 350]], np.int32)
    cv2.fillPoly(img, [pts], 0)
    
    # More circles
    cv2.circle(img, (300, 300), 40, 0, -1)
    cv2.circle(img, (500, 350), 60, 0, -1)
    
    print("\n1. Detecting shapes...")
    shapes_img = detect_shapes(img)
    
    print("\n2. Detecting circles...")
    circles_img, circles = detect_circles(img)
    
    print("\n3. Detecting contours...")
    contours_img, contours = contour_detection(img)
    
    print("\n4. Detecting blobs...")
    blobs_img, blobs = blob_detection(img)
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(cv2.cvtColor(shapes_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Shape Detection')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(circles_img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'Circle Detection ({len(circles[0]) if circles is not None else 0} circles)')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(cv2.cvtColor(contours_img, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'Contour Detection ({len(contours)} contours)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(blobs_img, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'Blob Detection ({len(blobs)} blobs)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n5. Object Detection Applications:")
    print("   - Face detection")
    print("   - Pedestrian detection")
    print("   - Vehicle detection")
    print("   - Product recognition")
    print("   - Quality inspection")
    print("   - Medical imaging")

if __name__ == "__main__":
    print(__doc__)
    demonstrate_object_detection()
    
    print("\n✓ Object detection complete!")
    print("Next: Learn about image segmentation in 06_image_segmentation.py")
