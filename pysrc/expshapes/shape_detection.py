import cv2
import numpy as np
import math

def calculate_angle(pt1, pt2, pt0):
    """
    Calculate angle between three points using dot product
    Equivalent to angle() function in C++ code
    """
    dx1 = pt1[0] - pt0[0]
    dy1 = pt1[1] - pt0[1]
    dx2 = pt2[0] - pt0[0]
    dy2 = pt2[1] - pt0[1]
    
    return (dx1*dx2 + dy1*dy2) / math.sqrt((dx1*dx1 + dy1*dy1) * (dx2*dx2 + dy2*dy2) + 1e-10)

def main():
    """
    Real-time rectangular shape detection and tracking using webcam input
    Equivalent to expshapes.cpp, working.cpp, and backupexpshapes.cpp
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not initialize camera")
        return -1
    
    # Create windows
    cv2.namedWindow("Original")
    cv2.namedWindow("HSV Thresholded")
    cv2.namedWindow("Morphological Operations")
    cv2.namedWindow("Shape Detection")
    
    print("Real-time rectangular shape detection started.")
    print("Detecting purple/blue rectangular objects...")
    print("Press ESC to exit, 'h' for help, 's' to save")
    
    frame_count = 0
    
    # HSV color range for purple/blue objects (from original C++ code)
    lower_hsv = np.array([150, 100, 50])   # cvScalar(150, 100, 50)
    upper_hsv = np.array([190, 255, 255])  # cvScalar(190, 255, 255)
    
    # Morphological kernels
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Apply HSV color thresholding
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        # Morphological operations (equivalent to original C++ sequence)
        # Opening: Erosion followed by Dilation (noise removal)
        eroded = cv2.erode(mask, erode_kernel, iterations=2)
        opened = cv2.dilate(eroded, dilate_kernel, iterations=2)
        
        # Closing: Dilation followed by Erosion (fill holes)
        dilated = cv2.dilate(opened, dilate_kernel, iterations=1)
        morphed = cv2.erode(dilated, erode_kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create display images
        shape_detection = frame.copy()
        morphed_colored = cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR)
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        rectangle_count = 0
        
        # Process each contour
        for contour in contours:
            # Calculate contour area and perimeter
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Filter by minimum area (equivalent to area > 20 in original)
            if area > 20 and perimeter > 0:
                # Approximate polygon (equivalent to cvApproxPoly)
                epsilon = 0.02 * perimeter  # 2% of perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's a 4-sided polygon (rectangle/quadrilateral)
                if len(approx) == 4:
                    rectangle_count += 1
                    
                    # Calculate moments for centroid (equivalent to cvMoments)
                    M = cv2.moments(contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                    else:
                        cx, cy = 0, 0
                    
                    # Draw rectangle edges (equivalent to cvLine)
                    for i in range(4):
                        pt1 = tuple(approx[i][0])
                        pt2 = tuple(approx[(i + 1) % 4][0])
                        cv2.line(shape_detection, pt1, pt2, (0, 255, 0), 2)
                    
                    # Draw contour
                    cv2.drawContours(shape_detection, [contour], -1, (255, 0, 0), 2)
                    
                    # Mark centroid
                    cv2.circle(shape_detection, (cx, cy), 5, (0, 0, 255), -1)
                    
                    # Add text label at centroid (equivalent to cvPutText)
                    cv2.putText(shape_detection, f"Rect {rectangle_count}", 
                               (cx - 30, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (255, 255, 255), 1)
                    
                    # Add area information
                    cv2.putText(shape_detection, f"Area: {int(area)}", 
                               (cx - 25, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.4, (255, 255, 255), 1)
                    
                    # Calculate and validate angles (similar to original angle checking)
                    angles_valid = True
                    for i in range(4):
                        pt1 = tuple(approx[i][0])
                        pt2 = tuple(approx[(i + 1) % 4][0])
                        pt0 = tuple(approx[(i + 3) % 4][0])
                        
                        cos_angle = abs(calculate_angle(pt1, pt2, pt0))
                        # Check if angle is close to 90 degrees (cosine ~ 0)
                        if cos_angle > 0.3:  # Not close to 90 degrees
                            angles_valid = False
                            break
                    
                    # Add angle validation indicator
                    if angles_valid:
                        cv2.putText(shape_detection, "Valid Rectangle", 
                                   (cx - 40, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.3, (0, 255, 0), 1)
                    else:
                        cv2.putText(shape_detection, "Invalid Angles", 
                                   (cx - 40, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.3, (0, 0, 255), 1)
        
        # Add frame information
        cv2.putText(shape_detection, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(shape_detection, f"Rectangles: {rectangle_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(shape_detection, f"Total Contours: {len(contours)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display all processing stages
        cv2.imshow("Original", frame)
        cv2.imshow("HSV Thresholded", mask_colored)
        cv2.imshow("Morphological Operations", morphed_colored)
        cv2.imshow("Shape Detection", shape_detection)
        
        # Print detection info
        if rectangle_count > 0:
            print(f"Frame {frame_count}: {rectangle_count} rectangle(s) detected")
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s'):  # Save current results
            cv2.imwrite(f"shapes_original_{frame_count}.jpg", frame)
            cv2.imwrite(f"shapes_mask_{frame_count}.jpg", mask)
            cv2.imwrite(f"shapes_morphed_{frame_count}.jpg", morphed)
            cv2.imwrite(f"shapes_detection_{frame_count}.jpg", shape_detection)
            print(f"Frame {frame_count} saved")
        elif key == ord('c'):  # Adjust HSV color range
            print(f"Current HSV range: {lower_hsv} to {upper_hsv}")
            print("Enter new HSV values:")
            try:
                h_min = int(input("H min (0-179): "))
                s_min = int(input("S min (0-255): "))
                v_min = int(input("V min (0-255): "))
                h_max = int(input("H max (0-179): "))
                s_max = int(input("S max (0-255): "))
                v_max = int(input("V max (0-255): "))
                
                lower_hsv = np.array([h_min, s_min, v_min])
                upper_hsv = np.array([h_max, s_max, v_max])
                print(f"HSV range updated: {lower_hsv} to {upper_hsv}")
            except ValueError:
                print("Invalid HSV values")
        elif key == ord('h'):  # Show help
            print("\nControls:")
            print("ESC - Exit")
            print("s - Save current frame and processing stages")
            print("c - Adjust HSV color range")
            print("h - Show this help")
            print("\nDetection Info:")
            print("- Target color: Purple/Blue objects")
            print("- HSV range: [150,100,50] to [190,255,255]")
            print("- Minimum area: 20 pixels")
            print("- Shape: 4-sided polygons (rectangles)")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total frames processed: {frame_count}")

def test_camera():
    """
    Simple camera test utility
    Equivalent to test.cpp
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not initialize camera")
        return -1
    
    # Create window (equivalent to "naren" window in original)
    cv2.namedWindow("Camera Test")
    
    print("Camera test started. Press ESC to exit.")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break
        
        frame_count += 1
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Camera Test", frame)
        
        # Check for ESC key (ASCII 27)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Camera test completed. Total frames: {frame_count}")

def detect_shapes_in_image(image_path):
    """
    Detect rectangular shapes in a static image
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    
    # HSV color range
    lower_hsv = np.array([150, 100, 50])
    upper_hsv = np.array([190, 255, 255])
    
    # Convert and threshold
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(mask, kernel, iterations=2)
    opened = cv2.dilate(eroded, kernel, iterations=2)
    dilated = cv2.dilate(opened, kernel, iterations=1)
    morphed = cv2.erode(dilated, kernel, iterations=1)
    
    # Find contours and detect rectangles
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result = img.copy()
    rectangle_count = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if area > 20 and perimeter > 0:
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                rectangle_count += 1
                
                # Draw rectangle
                for i in range(4):
                    pt1 = tuple(approx[i][0])
                    pt2 = tuple(approx[(i + 1) % 4][0])
                    cv2.line(result, pt1, pt2, (0, 255, 0), 2)
                
                # Calculate and mark centroid
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(result, f"Rect {rectangle_count}", 
                               (cx - 30, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (255, 255, 255), 2)
    
    print(f"Detected {rectangle_count} rectangle(s)")
    
    # Display results
    cv2.namedWindow("Original", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Mask", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Detection Result", cv2.WINDOW_AUTOSIZE)
    
    cv2.imshow("Original", img)
    cv2.imshow("Mask", mask)
    cv2.imshow("Detection Result", result)
    
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # Real-time shape detection mode
        main()
    elif sys.argv[1] == "test":
        # Camera test mode
        test_camera()
    else:
        # Static image mode
        detect_shapes_in_image(sys.argv[1])