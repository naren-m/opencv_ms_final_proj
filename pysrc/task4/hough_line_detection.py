import cv2
import numpy as np
import math

def main():
    """
    Real-time line detection using Canny edge detection followed by Hough line transformation
    Equivalent to houghLines.cpp
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not initialize camera")
        return -1
    
    cv2.namedWindow("Original")
    cv2.namedWindow("Edges")
    cv2.namedWindow("Line Detection")
    cv2.namedWindow("Combined")
    
    print("Real-time line detection started.")
    print("Press ESC to exit, 's' to save, 'h' for help")
    
    frame_count = 0
    
    # Hough line parameters (adjustable)
    rho = 1              # Distance resolution in pixels
    theta = np.pi/180    # Angle resolution in radians
    threshold = 50       # Minimum number of votes
    min_line_length = 50 # Minimum line length
    max_line_gap = 10    # Maximum gap between line segments
    
    # Canny parameters
    canny_low = 50
    canny_high = 150
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, canny_low, canny_high)
        
        # Apply probabilistic Hough line detection
        lines = cv2.HoughLinesP(
            edges,
            rho=rho,
            theta=theta,
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )
        
        # Create image for line visualization
        line_img = np.zeros_like(frame)
        line_count = 0
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate line length
                length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                # Filter out very short lines
                if length > min_line_length:
                    # Draw line
                    cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Mark line endpoints
                    cv2.circle(line_img, (x1, y1), 3, (255, 0, 0), -1)
                    cv2.circle(line_img, (x2, y2), 3, (0, 0, 255), -1)
                    
                    line_count += 1
        
        # Create combined visualization
        combined = cv2.addWeighted(frame, 0.7, line_img, 0.3, 0)
        
        # Add text overlays
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Lines: {line_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Threshold: {threshold}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(line_img, "Detected Lines", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display all images
        cv2.imshow("Original", frame)
        cv2.imshow("Edges", edges)
        cv2.imshow("Line Detection", line_img)
        cv2.imshow("Combined", combined)
        
        # Print line detection info
        if line_count > 0:
            print(f"Frame {frame_count}: {line_count} lines detected")
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s'):  # Save current results
            cv2.imwrite(f"hough_original_{frame_count}.jpg", frame)
            cv2.imwrite(f"hough_edges_{frame_count}.jpg", edges)
            cv2.imwrite(f"hough_lines_{frame_count}.jpg", line_img)
            cv2.imwrite(f"hough_combined_{frame_count}.jpg", combined)
            print(f"Frame {frame_count} saved")
        elif key == ord('t'):  # Adjust threshold
            print(f"Current threshold: {threshold}")
            try:
                new_threshold = int(input("Enter new threshold (10-200): "))
                threshold = max(10, min(200, new_threshold))
                print(f"Threshold set to: {threshold}")
            except ValueError:
                print("Invalid threshold value")
        elif key == ord('c'):  # Adjust Canny parameters
            print(f"Current Canny: Low={canny_low}, High={canny_high}")
            try:
                new_low = int(input("Enter new low threshold (0-255): "))
                new_high = int(input("Enter new high threshold (0-255): "))
                canny_low = max(0, min(255, new_low))
                canny_high = max(0, min(255, new_high))
                print(f"Canny set to: Low={canny_low}, High={canny_high}")
            except ValueError:
                print("Invalid Canny values")
        elif key == ord('l'):  # Adjust line length
            print(f"Current min line length: {min_line_length}")
            try:
                new_length = int(input("Enter new minimum line length: "))
                min_line_length = max(10, new_length)
                print(f"Minimum line length set to: {min_line_length}")
            except ValueError:
                print("Invalid line length value")
        elif key == ord('h'):  # Show help
            print("\nControls:")
            print("ESC - Exit")
            print("s - Save current frame and results")
            print("t - Adjust Hough threshold")
            print("c - Adjust Canny parameters")
            print("l - Adjust minimum line length")
            print("h - Show this help")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total frames processed: {frame_count}")

def standard_hough_lines(image):
    """
    Standard Hough line detection (equivalent to commented code in original)
    Returns lines in polar coordinate form (rho, theta)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Standard Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    line_img = np.zeros_like(image)
    
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            
            # Calculate line endpoints
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return line_img

def detect_lines_in_image(image_path):
    """
    Detect lines in a static image using both standard and probabilistic Hough transforms
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    
    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Probabilistic Hough lines
    lines_p = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                              minLineLength=50, maxLineGap=10)
    
    # Standard Hough lines
    lines_standard = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    # Create visualizations
    img_prob = img.copy()
    img_standard = img.copy()
    
    # Draw probabilistic lines
    if lines_p is not None:
        for line in lines_p:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_prob, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw standard lines
    if lines_standard is not None:
        for line in lines_standard:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            
            cv2.line(img_standard, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Display results
    cv2.namedWindow("Original", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Edges", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Probabilistic Hough", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Standard Hough", cv2.WINDOW_AUTOSIZE)
    
    cv2.imshow("Original", img)
    cv2.imshow("Edges", edges)
    cv2.imshow("Probabilistic Hough", img_prob)
    cv2.imshow("Standard Hough", img_standard)
    
    prob_count = len(lines_p) if lines_p is not None else 0
    standard_count = len(lines_standard) if lines_standard is not None else 0
    
    print(f"Probabilistic Hough: {prob_count} lines detected")
    print(f"Standard Hough: {standard_count} lines detected")
    print("Press any key to exit...")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def compare_hough_methods(image_path):
    """
    Compare different Hough line detection methods
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Different parameter sets for comparison
    methods = [
        {"name": "Low Threshold", "threshold": 30, "minLineLength": 30, "maxLineGap": 15},
        {"name": "Medium Threshold", "threshold": 50, "minLineLength": 50, "maxLineGap": 10},
        {"name": "High Threshold", "threshold": 100, "minLineLength": 80, "maxLineGap": 5},
        {"name": "Relaxed", "threshold": 20, "minLineLength": 20, "maxLineGap": 20}
    ]
    
    results = []
    
    for method in methods:
        img_copy = img.copy()
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                               threshold=method["threshold"],
                               minLineLength=method["minLineLength"],
                               maxLineGap=method["maxLineGap"])
        
        line_count = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                line_count += 1
        
        cv2.putText(img_copy, f"{method['name']}: {line_count} lines", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        results.append(img_copy)
        print(f"{method['name']}: {line_count} lines detected")
    
    # Display results in grid
    top_row = np.hstack([results[0], results[1]])
    bottom_row = np.hstack([results[2], results[3]])
    comparison = np.vstack([top_row, bottom_row])
    
    cv2.namedWindow("Hough Methods Comparison", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Hough Methods Comparison", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # Real-time camera mode
        main()
    elif len(sys.argv) == 2:
        # Static image mode
        detect_lines_in_image(sys.argv[1])
    elif len(sys.argv) == 3 and sys.argv[1] == "compare":
        # Comparison mode
        compare_hough_methods(sys.argv[2])
    else:
        print("Usage:")
        print("  python hough_line_detection.py                    # Camera mode")
        print("  python hough_line_detection.py <image_path>       # Single image mode")
        print("  python hough_line_detection.py compare <image_path>  # Compare methods")