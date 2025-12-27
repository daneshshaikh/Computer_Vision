"""
COMPUTER VISION - PRACTICAL PROJECTS
=====================================

Hands-on projects to apply computer vision concepts.

Projects:
1. Face Detection System
2. Motion Detection
3. Color-based Object Tracker
4. Document Scanner
5. Barcode/QR Code Reader
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Project 1: Face Detection using Haar Cascades
class FaceDetector:
    """Simple face detection system"""
    
    def __init__(self):
        # Load pre-trained Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
    
    def detect_faces(self, img):
        """Detect faces in an image"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            # Detect eyes within face region
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        return img, len(faces)

# Project 2: Motion Detection
class MotionDetector:
    """Detect motion between frames"""
    
    def __init__(self, threshold=25):
        self.threshold = threshold
        self.previous_frame = None
    
    def detect_motion(self, frame):
        """Detect motion in current frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.previous_frame is None:
            self.previous_frame = gray
            return frame, False
        
        # Compute difference
        frame_delta = cv2.absdiff(self.previous_frame, gray)
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                motion_detected = True
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        self.previous_frame = gray
        return frame, motion_detected

# Project 3: Color-based Object Tracker
class ColorTracker:
    """Track objects based on color"""
    
    def __init__(self, lower_color, upper_color):
        self.lower_color = np.array(lower_color)
        self.upper_color = np.array(upper_color)
    
    def track(self, frame):
        """Track colored object in frame"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) > 500:
                # Get bounding circle
                ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
                
                if radius > 10:
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    return frame, (int(x), int(y))
        
        return frame, None

# Project 4: Document Scanner
class DocumentScanner:
    """Scan and perspective-correct documents"""
    
    def order_points(self, pts):
        """Order points in clockwise order"""
        rect = np.zeros((4, 2), dtype="float32")
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def four_point_transform(self, image, pts):
        """Apply perspective transform"""
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        
        # Compute width
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        # Compute height
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # Destination points
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        
        # Perspective transform
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        return warped
    
    def scan(self, image):
        """Scan document from image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        # Find document contour
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4:
                # Apply perspective transform
                warped = self.four_point_transform(image, approx.reshape(4, 2))
                return warped
        
        return image

# Demonstration
def demonstrate_projects():
    """Demonstrate practical projects"""
    print("Computer Vision Practical Projects")
    print("="*50)
    
    print("\n1. Face Detection System")
    print("   - Detects faces and eyes in images")
    print("   - Uses Haar Cascade classifiers")
    print("   - Real-time capable")
    
    print("\n2. Motion Detection")
    print("   - Detects movement between frames")
    print("   - Useful for security systems")
    print("   - Background subtraction technique")
    
    print("\n3. Color-based Object Tracker")
    print("   - Tracks objects by color")
    print("   - HSV color space")
    print("   - Real-time tracking")
    
    print("\n4. Document Scanner")
    print("   - Detects document edges")
    print("   - Applies perspective correction")
    print("   - Creates scanned image")
    
    print("\n5. Additional Project Ideas:")
    print("   - Lane detection for self-driving cars")
    print("   - Hand gesture recognition")
    print("   - Optical Character Recognition (OCR)")
    print("   - Image stitching for panoramas")
    print("   - Real-time filters (Instagram-like)")
    print("   - People counter")
    print("   - Parking space detector")

if __name__ == "__main__":
    print(__doc__)
    demonstrate_projects()
    
    print("\n\n" + "="*50)
    print("✓ Practical projects overview complete!")
    print("="*50)
    print("\nTo run these projects:")
    print("  1. Instantiate the class")
    print("  2. Load or capture images/video")
    print("  3. Call the appropriate methods")
    print("\nSee README.md for detailed usage examples")
