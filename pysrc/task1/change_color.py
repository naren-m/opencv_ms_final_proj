import cv2
import numpy as np

def main():
    """
    Real-time video capture with HSV color space conversion and channel manipulation
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not initialize camera")
        return -1
    
    # Create windows
    cv2.namedWindow("Original")
    cv2.namedWindow("HSV")
    cv2.namedWindow("Hue")
    cv2.namedWindow("Saturation") 
    cv2.namedWindow("Value")
    cv2.namedWindow("Normalized HSV")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Split HSV channels
        h, s, v = cv2.split(hsv_frame)
        
        # Create single channel display images (convert to 3-channel for display)
        hue_display = cv2.merge([h, h, h])
        sat_display = cv2.merge([s, s, s])
        val_display = cv2.merge([v, v, v])
        
        # Normalize HSV to 0-1 range and convert back for display
        hsv_normalized = hsv_frame.astype(np.float32) / 255.0
        hsv_display = (hsv_normalized * 255).astype(np.uint8)
        
        # Display all images
        cv2.imshow("Original", frame)
        cv2.imshow("HSV", hsv_frame)
        cv2.imshow("Hue", hue_display)
        cv2.imshow("Saturation", sat_display)
        cv2.imshow("Value", val_display)
        cv2.imshow("Normalized HSV", hsv_display)
        
        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()