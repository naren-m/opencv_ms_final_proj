import cv2
import numpy as np

# Global variables for trackbar
canny_low = 50
canny_high = 150

def update_canny_low(val):
    """Callback for low threshold trackbar"""
    global canny_low
    canny_low = val

def update_canny_high(val):
    """Callback for high threshold trackbar"""
    global canny_high
    canny_high = val

def main():
    """
    Real-time Canny edge detection with interactive threshold control
    Equivalent to cannyTest.cpp
    """
    global canny_low, canny_high
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not initialize camera")
        return -1
    
    # Create windows
    cv2.namedWindow("Original")
    cv2.namedWindow("Grayscale")
    cv2.namedWindow("Blurred")
    cv2.namedWindow("Canny Edges")
    cv2.namedWindow("Edge Overlay")
    cv2.namedWindow("Controls")
    
    # Create trackbars for threshold control
    cv2.createTrackbar("Low Threshold", "Controls", canny_low, 255, update_canny_low)
    cv2.createTrackbar("High Threshold", "Controls", canny_high, 255, update_canny_high)
    
    # Create a black image for controls window
    controls_img = np.zeros((100, 400, 3), dtype=np.uint8)
    cv2.imshow("Controls", controls_img)
    
    print("Canny edge detection started.")
    print("Use trackbars to adjust thresholds. Press ESC to exit.")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply blur to reduce noise (3x3 kernel as in original)
        blurred = cv2.blur(gray, (3, 3))
        
        # Apply Canny edge detection with current threshold values
        edges = cv2.Canny(blurred, canny_low, canny_high)
        
        # Create colored edge overlay (similar to original copyTo operation)
        edge_overlay = frame.copy()
        edge_overlay[edges == 255] = [0, 255, 0]  # Green edges
        
        # Create edge visualization on original image
        edge_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Add text overlays
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(edge_overlay, f"Low: {canny_low}, High: {canny_high}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display all images
        cv2.imshow("Original", frame)
        cv2.imshow("Grayscale", gray)
        cv2.imshow("Blurred", blurred)
        cv2.imshow("Canny Edges", edges)
        cv2.imshow("Edge Overlay", edge_overlay)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s'):  # Save current results
            cv2.imwrite(f"canny_original_{frame_count}.jpg", frame)
            cv2.imwrite(f"canny_edges_{frame_count}.jpg", edges)
            cv2.imwrite(f"canny_overlay_{frame_count}.jpg", edge_overlay)
            print(f"Frame {frame_count} saved with thresholds: Low={canny_low}, High={canny_high}")
        elif key == ord('r'):  # Reset thresholds to default
            canny_low = 50
            canny_high = 150
            cv2.setTrackbarPos("Low Threshold", "Controls", canny_low)
            cv2.setTrackbarPos("High Threshold", "Controls", canny_high)
            print("Thresholds reset to default")
        elif key == ord('a'):  # Auto-adjust thresholds (1:3 ratio)
            canny_high = max(100, canny_low * 3)
            cv2.setTrackbarPos("High Threshold", "Controls", min(255, canny_high))
            print(f"Auto-adjusted: Low={canny_low}, High={canny_high}")
        elif key == ord('h'):  # Show help
            print("\nControls:")
            print("ESC - Exit")
            print("s - Save current frame and edges")
            print("r - Reset thresholds to default")
            print("a - Auto-adjust high threshold (3x low threshold)")
            print("h - Show this help")
            print("Use trackbars to adjust thresholds manually")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total frames processed: {frame_count}")

def canny_on_image(image_path):
    """
    Apply Canny edge detection to a static image with interactive controls
    """
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Create windows and trackbars
    cv2.namedWindow("Original Image")
    cv2.namedWindow("Canny Edges")
    cv2.namedWindow("Controls")
    
    cv2.createTrackbar("Low Threshold", "Controls", canny_low, 255, update_canny_low)
    cv2.createTrackbar("High Threshold", "Controls", canny_high, 255, update_canny_high)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(gray, (3, 3))
    
    print(f"Processing image: {image_path}")
    print("Adjust thresholds with trackbars. Press any key to exit.")
    
    while True:
        # Apply Canny with current thresholds
        edges = cv2.Canny(blurred, canny_low, canny_high)
        
        # Create display images
        img_display = img.copy()
        cv2.putText(img_display, f"Low: {canny_low}, High: {canny_high}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show images
        cv2.imshow("Original Image", img_display)
        cv2.imshow("Canny Edges", edges)
        
        # Create controls display
        controls_img = np.zeros((100, 400, 3), dtype=np.uint8)
        cv2.putText(controls_img, f"Low: {canny_low}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(controls_img, f"High: {canny_high}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Controls", controls_img)
        
        if cv2.waitKey(30) & 0xFF != 255:  # Exit on any key press
            break
    
    cv2.destroyAllWindows()

def batch_canny_processing(input_dir, output_dir, low_thresh=50, high_thresh=150):
    """
    Process multiple images with Canny edge detection
    """
    import os
    import glob
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Supported image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    print(f"Found {len(image_files)} images to process")
    
    for i, img_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
        
        # Load and process image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load: {img_path}")
            continue
        
        # Convert to grayscale and apply Canny
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.blur(gray, (3, 3))
        edges = cv2.Canny(blurred, low_thresh, high_thresh)
        
        # Save result
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_canny.jpg")
        cv2.imwrite(output_path, edges)
    
    print(f"Batch processing complete. Results saved to: {output_dir}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # Real-time camera mode
        main()
    elif len(sys.argv) == 2:
        # Static image mode
        canny_on_image(sys.argv[1])
    elif len(sys.argv) >= 3 and sys.argv[1] == "batch":
        # Batch processing mode
        input_dir = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "canny_output"
        low_thresh = int(sys.argv[4]) if len(sys.argv) > 4 else 50
        high_thresh = int(sys.argv[5]) if len(sys.argv) > 5 else 150
        batch_canny_processing(input_dir, output_dir, low_thresh, high_thresh)
    else:
        print("Usage:")
        print("  python canny_edge_detection.py                    # Camera mode")
        print("  python canny_edge_detection.py <image_path>       # Single image mode")
        print("  python canny_edge_detection.py batch <input_dir> [output_dir] [low] [high]  # Batch mode")