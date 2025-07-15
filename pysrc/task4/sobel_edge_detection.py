import cv2
import numpy as np

def main():
    """
    Sobel edge detection on static image or camera feed
    Equivalent to sober.cpp (note: original filename appears to be "sober" not "sobel")
    """
    import sys
    
    if len(sys.argv) > 1:
        # Static image mode
        process_static_image(sys.argv[1])
    else:
        # Real-time camera mode
        process_camera_feed()

def process_static_image(image_path):
    """
    Apply Sobel edge detection to a static image
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    
    # Apply median blur to reduce noise (as in original C++ code)
    blurred = cv2.medianBlur(img, 5)
    
    # Convert to grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    # Apply Sobel edge detection in X and Y directions
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # X gradient
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Y gradient
    
    # Convert to absolute values and scale to 0-255 range
    sobel_x_abs = cv2.convertScaleAbs(sobel_x)
    sobel_y_abs = cv2.convertScaleAbs(sobel_y)
    
    # Combine X and Y gradients (equivalent to addWeighted in original)
    sobel_combined = cv2.addWeighted(sobel_x_abs, 0.5, sobel_y_abs, 0.5, 0)
    
    # Alternative combination methods
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_magnitude = cv2.convertScaleAbs(sobel_magnitude)
    
    # Create directional visualization
    sobel_direction = np.arctan2(sobel_y, sobel_x)
    sobel_direction_vis = cv2.convertScaleAbs(sobel_direction * 180 / np.pi)
    
    # Display results
    cv2.namedWindow("Original", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Blurred", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Sobel X", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Sobel Y", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Sobel Combined", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Sobel Magnitude", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Edge Direction", cv2.WINDOW_AUTOSIZE)
    
    cv2.imshow("Original", img)
    cv2.imshow("Blurred", blurred)
    cv2.imshow("Sobel X", sobel_x_abs)
    cv2.imshow("Sobel Y", sobel_y_abs)
    cv2.imshow("Sobel Combined", sobel_combined)
    cv2.imshow("Sobel Magnitude", sobel_magnitude)
    cv2.imshow("Edge Direction", sobel_direction_vis)
    
    print("Sobel edge detection complete.")
    print("Press 's' to save results, any other key to exit...")
    
    key = cv2.waitKey(0)
    if key == ord('s'):
        # Save results
        base_name = image_path.split('.')[0]
        cv2.imwrite(f"{base_name}_sobel_x.jpg", sobel_x_abs)
        cv2.imwrite(f"{base_name}_sobel_y.jpg", sobel_y_abs)
        cv2.imwrite(f"{base_name}_sobel_combined.jpg", sobel_combined)
        cv2.imwrite(f"{base_name}_sobel_magnitude.jpg", sobel_magnitude)
        print("Results saved")
    
    cv2.destroyAllWindows()

def process_camera_feed():
    """
    Real-time Sobel edge detection from camera
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not initialize camera")
        return -1
    
    cv2.namedWindow("Original")
    cv2.namedWindow("Sobel X")
    cv2.namedWindow("Sobel Y")
    cv2.namedWindow("Sobel Combined")
    cv2.namedWindow("Sobel Magnitude")
    
    print("Real-time Sobel edge detection started.")
    print("Press ESC to exit, 's' to save current frame, 'k' to change kernel size")
    
    frame_count = 0
    kernel_size = 3  # Sobel kernel size (3, 5, or 7)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Apply median blur
        blurred = cv2.medianBlur(frame, 5)
        
        # Convert to grayscale
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        
        # Apply Sobel edge detection
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
        
        # Convert to absolute values
        sobel_x_abs = cv2.convertScaleAbs(sobel_x)
        sobel_y_abs = cv2.convertScaleAbs(sobel_y)
        
        # Combine gradients
        sobel_combined = cv2.addWeighted(sobel_x_abs, 0.5, sobel_y_abs, 0.5, 0)
        
        # Calculate magnitude
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_magnitude = cv2.convertScaleAbs(sobel_magnitude)
        
        # Add text overlays
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Kernel: {kernel_size}x{kernel_size}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(sobel_combined, "Combined Gradients", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display all results
        cv2.imshow("Original", frame)
        cv2.imshow("Sobel X", sobel_x_abs)
        cv2.imshow("Sobel Y", sobel_y_abs)
        cv2.imshow("Sobel Combined", sobel_combined)
        cv2.imshow("Sobel Magnitude", sobel_magnitude)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s'):  # Save current frame
            cv2.imwrite(f"sobel_original_{frame_count}.jpg", frame)
            cv2.imwrite(f"sobel_x_{frame_count}.jpg", sobel_x_abs)
            cv2.imwrite(f"sobel_y_{frame_count}.jpg", sobel_y_abs)
            cv2.imwrite(f"sobel_combined_{frame_count}.jpg", sobel_combined)
            cv2.imwrite(f"sobel_magnitude_{frame_count}.jpg", sobel_magnitude)
            print(f"Frame {frame_count} saved")
        elif key == ord('k'):  # Change kernel size
            if kernel_size == 3:
                kernel_size = 5
            elif kernel_size == 5:
                kernel_size = 7
            else:
                kernel_size = 3
            print(f"Kernel size changed to {kernel_size}x{kernel_size}")
        elif key == ord('h'):  # Show help
            print("\nControls:")
            print("ESC - Exit")
            print("s - Save current frame and edge results")
            print("k - Change Sobel kernel size (3, 5, 7)")
            print("h - Show this help")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total frames processed: {frame_count}")

def compare_edge_detectors(image_path):
    """
    Compare Sobel with other edge detection methods
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Preprocess
    blurred = cv2.medianBlur(img, 5)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    # Sobel edge detection
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.convertScaleAbs(cv2.addWeighted(
        cv2.convertScaleAbs(sobel_x), 0.5, 
        cv2.convertScaleAbs(sobel_y), 0.5, 0))
    
    # Canny edge detection
    canny = cv2.Canny(gray, 50, 150)
    
    # Laplacian edge detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    
    # Scharr edge detection (improved Sobel)
    scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    scharr_combined = cv2.convertScaleAbs(cv2.addWeighted(
        cv2.convertScaleAbs(scharr_x), 0.5,
        cv2.convertScaleAbs(scharr_y), 0.5, 0))
    
    # Create comparison display
    results = [
        ("Original", img),
        ("Sobel", cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2BGR)),
        ("Canny", cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)),
        ("Laplacian", cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)),
        ("Scharr", cv2.cvtColor(scharr_combined, cv2.COLOR_GRAY2BGR))
    ]
    
    # Resize images for display if needed
    display_height = 300
    for i, (name, result_img) in enumerate(results):
        height, width = result_img.shape[:2]
        if height > display_height:
            scale = display_height / height
            new_width = int(width * scale)
            result_img = cv2.resize(result_img, (new_width, display_height))
            results[i] = (name, result_img)
        
        # Add labels
        cv2.putText(result_img, name, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Create grid display
    if len(results) >= 4:
        top_row = np.hstack([results[0][1], results[1][1]])
        middle_row = np.hstack([results[2][1], results[3][1]])
        if len(results) == 5:
            bottom_row = np.hstack([results[4][1], np.zeros_like(results[4][1])])
            comparison = np.vstack([top_row, middle_row, bottom_row])
        else:
            comparison = np.vstack([top_row, middle_row])
    else:
        comparison = np.hstack([result[1] for result in results])
    
    cv2.namedWindow("Edge Detection Comparison", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Edge Detection Comparison", comparison)
    
    print("Edge detection comparison complete.")
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # Real-time camera mode
        main()
    elif len(sys.argv) == 2:
        # Static image mode
        main()
    elif len(sys.argv) == 3 and sys.argv[1] == "compare":
        # Comparison mode
        compare_edge_detectors(sys.argv[2])
    else:
        print("Usage:")
        print("  python sobel_edge_detection.py                     # Camera mode")
        print("  python sobel_edge_detection.py <image_path>        # Single image mode")
        print("  python sobel_edge_detection.py compare <image_path>   # Compare methods")