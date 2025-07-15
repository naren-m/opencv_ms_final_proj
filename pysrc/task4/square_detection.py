import cv2
import numpy as np
import math

def calculate_angle(pt1, pt2, pt0):
    """
    Calculate angle between three points (equivalent to angle function in C++)
    """
    dx1 = pt1[0] - pt0[0]
    dy1 = pt1[1] - pt0[1]
    dx2 = pt2[0] - pt0[0]
    dy2 = pt2[1] - pt0[1]
    
    return (dx1*dx2 + dy1*dy2) / math.sqrt((dx1*dx1 + dy1*dy1) * (dx2*dx2 + dy2*dy2) + 1e-10)

def find_squares(image, min_area=1000):
    """
    Find squares in an image using contour detection and geometric validation
    Equivalent to findSquares function in C++
    """
    squares = []
    
    # Create multiple processed versions for better detection
    processed_images = []
    
    # 1. Apply dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(image, kernel, iterations=1)
    processed_images.append(dilated)
    
    # 2. Apply median blur to reduce noise
    blurred = cv2.medianBlur(image, 9)
    processed_images.append(blurred)
    
    # 3. Pyramid down and up for noise filtering
    pyr_down = cv2.pyrDown(image)
    pyr_up = cv2.pyrUp(pyr_down)
    if pyr_up.shape[:2] != image.shape[:2]:
        pyr_up = cv2.resize(pyr_up, (image.shape[1], image.shape[0]))
    processed_images.append(pyr_up)
    
    # Find squares in each processed image
    for processed in processed_images:
        # Convert to different color channels if image is colored
        if len(processed.shape) == 3:
            # Split channels and process each one
            channels = cv2.split(processed)
        else:
            channels = [processed]
        
        for channel in channels:
            # Apply Canny edge detection
            edges = cv2.Canny(channel, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if polygon has 4 vertices and is convex
                if (len(approx) == 4 and 
                    cv2.isContourConvex(approx) and 
                    cv2.contourArea(approx) > min_area):
                    
                    # Convert to list of points
                    points = [tuple(point[0]) for point in approx]
                    
                    # Check if angles are approximately 90 degrees
                    angles_ok = True
                    for i in range(4):
                        pt1 = points[i]
                        pt2 = points[(i + 1) % 4]
                        pt0 = points[(i + 3) % 4]
                        
                        cos_angle = abs(calculate_angle(pt1, pt2, pt0))
                        if cos_angle > 0.3:  # Angle should be close to 90 degrees
                            angles_ok = False
                            break
                    
                    if angles_ok:
                        # Check if square is not already found (avoid duplicates)
                        is_duplicate = False
                        for existing_square in squares:
                            # Simple distance check for duplicate detection
                            if all(min(math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) 
                                      for p2 in existing_square) < 20 for p1 in points):
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            squares.append(points)
    
    return squares

def main():
    """
    Real-time square detection from webcam
    Equivalent to working.cpp and copyworking.cpp
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not initialize camera")
        return -1
    
    cv2.namedWindow("Square Detection")
    cv2.namedWindow("Processed")
    
    print("Real-time square detection started.")
    print("Press ESC to exit, 's' to save current frame, 'a' to adjust parameters")
    
    frame_count = 0
    min_area = 1000  # Minimum area for square detection
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Create a copy for drawing
        display_frame = frame.copy()
        
        # Find squares in the frame
        squares = find_squares(frame, min_area)
        
        # Draw detected squares
        for i, square in enumerate(squares):
            # Convert points to numpy array for drawing
            pts = np.array(square, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Draw square outline
            cv2.polylines(display_frame, [pts], True, (0, 255, 0), 3)
            
            # Draw corner points
            for point in square:
                cv2.circle(display_frame, point, 5, (0, 0, 255), -1)
            
            # Add square label
            center_x = sum(p[0] for p in square) // 4
            center_y = sum(p[1] for p in square) // 4
            cv2.putText(display_frame, f"Square {i+1}", (center_x-30, center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Calculate and display area
            area = cv2.contourArea(pts)
            cv2.putText(display_frame, f"Area: {int(area)}", (center_x-30, center_y+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Add info overlay
        cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Squares: {len(squares)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Min Area: {min_area}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Create processed view for debugging
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        processed_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Display results
        cv2.imshow("Square Detection", display_frame)
        cv2.imshow("Processed", processed_display)
        
        # Print detection info
        if len(squares) > 0:
            print(f"Frame {frame_count}: {len(squares)} square(s) detected")
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s'):  # Save current frame
            cv2.imwrite(f"square_detection_{frame_count}.jpg", display_frame)
            cv2.imwrite(f"square_edges_{frame_count}.jpg", edges)
            print(f"Frame {frame_count} saved")
        elif key == ord('a'):  # Adjust minimum area
            print(f"Current minimum area: {min_area}")
            try:
                new_area = int(input("Enter new minimum area: "))
                min_area = max(100, new_area)
                print(f"Minimum area set to: {min_area}")
            except ValueError:
                print("Invalid area value")
        elif key == ord('h'):  # Show help
            print("\nControls:")
            print("ESC - Exit")
            print("s - Save current frame")
            print("a - Adjust minimum area threshold")
            print("h - Show this help")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total frames processed: {frame_count}")

def detect_squares_in_image(image_path):
    """
    Detect squares in a static image
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    
    # Find squares
    squares = find_squares(img)
    
    # Draw results
    result = img.copy()
    for i, square in enumerate(squares):
        pts = np.array(square, np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        cv2.polylines(result, [pts], True, (0, 255, 0), 3)
        
        for point in square:
            cv2.circle(result, point, 5, (0, 0, 255), -1)
        
        # Add label
        center_x = sum(p[0] for p in square) // 4
        center_y = sum(p[1] for p in square) // 4
        cv2.putText(result, f"Square {i+1}", (center_x-30, center_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    print(f"Detected {len(squares)} square(s)")
    
    # Display results
    cv2.namedWindow("Original", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Square Detection", cv2.WINDOW_AUTOSIZE)
    
    cv2.imshow("Original", img)
    cv2.imshow("Square Detection", result)
    
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_square_detection():
    """
    Test square detection with synthetic images
    """
    # Create test image with squares
    test_img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Draw some squares
    cv2.rectangle(test_img, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.rectangle(test_img, (200, 200), (350, 350), (128, 128, 128), -1)
    cv2.rectangle(test_img, (100, 250), (180, 330), (200, 200, 200), -1)
    
    # Add some noise
    noise = np.random.randint(0, 50, test_img.shape, dtype=np.uint8)
    test_img = cv2.add(test_img, noise)
    
    print("Testing square detection on synthetic image...")
    
    # Detect squares
    squares = find_squares(test_img)
    
    # Draw results
    result = test_img.copy()
    for i, square in enumerate(squares):
        pts = np.array(square, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(result, [pts], True, (0, 255, 0), 2)
        
        center_x = sum(p[0] for p in square) // 4
        center_y = sum(p[1] for p in square) // 4
        cv2.putText(result, f"S{i+1}", (center_x, center_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    print(f"Test result: {len(squares)} square(s) detected")
    
    cv2.imshow("Test Image", test_img)
    cv2.imshow("Detection Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # Real-time camera mode
        main()
    elif sys.argv[1] == "test":
        # Test mode
        test_square_detection()
    else:
        # Static image mode
        detect_squares_in_image(sys.argv[1])