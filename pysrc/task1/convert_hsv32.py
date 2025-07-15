import cv2
import numpy as np

def main():
    """
    HSV color space conversion with channel visualization (32-bit depth)
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not initialize camera")
        return -1
    
    cv2.namedWindow("Original")
    cv2.namedWindow("HSV") 
    cv2.namedWindow("Hue Channel")
    cv2.namedWindow("Saturation Channel")
    cv2.namedWindow("Value Channel")
    cv2.namedWindow("Normalized HSV")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to HSV with 32-bit precision
        hsv_32f = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_32f = hsv_32f.astype(np.float32)
        
        # Split channels
        h, s, v = cv2.split(hsv_32f)
        
        # Normalize channels to 0-1 range for visualization
        h_norm = h / 180.0  # Hue is 0-180 in OpenCV
        s_norm = s / 255.0  # Saturation is 0-255
        v_norm = v / 255.0  # Value is 0-255
        
        # Convert normalized channels back to 8-bit for display
        h_display = (h_norm * 255).astype(np.uint8)
        s_display = (s_norm * 255).astype(np.uint8)
        v_display = (v_norm * 255).astype(np.uint8)
        
        # Create colored channel visualizations
        h_colored = cv2.applyColorMap(h_display, cv2.COLORMAP_HSV)
        s_colored = cv2.merge([s_display, s_display, s_display])
        v_colored = cv2.merge([v_display, v_display, v_display])
        
        # Create normalized HSV image
        hsv_normalized = np.stack([h_norm, s_norm, v_norm], axis=2)
        hsv_display = (hsv_normalized * 255).astype(np.uint8)
        
        # Display original HSV (8-bit)
        hsv_8bit = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Display all images
        cv2.imshow("Original", frame)
        cv2.imshow("HSV", hsv_8bit)
        cv2.imshow("Hue Channel", h_colored)
        cv2.imshow("Saturation Channel", s_colored)
        cv2.imshow("Value Channel", v_colored)
        cv2.imshow("Normalized HSV", hsv_display)
        
        # Print some pixel values for debugging
        height, width = frame.shape[:2]
        center_y, center_x = height // 2, width // 2
        
        if cv2.waitKey(1) & 0xFF == ord('p'):  # Press 'p' to print values
            print(f"\nCenter pixel ({center_x}, {center_y}):")
            print(f"Original BGR: {frame[center_y, center_x]}")
            print(f"HSV 8-bit: {hsv_8bit[center_y, center_x]}")
            print(f"HSV 32-bit: [{h[center_y, center_x]:.2f}, {s[center_y, center_x]:.2f}, {v[center_y, center_x]:.2f}]")
            print(f"Normalized: [{h_norm[center_y, center_x]:.4f}, {s_norm[center_y, center_x]:.4f}, {v_norm[center_y, center_x]:.4f}]")
        
        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()