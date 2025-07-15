import cv2
import numpy as np
import sys
import os

def get_thresholded_image(img):
    """
    Convert BGR image to HSV and apply threshold for yellow object detection
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # HSV threshold for yellow objects (similar to original C++ ranges)
    lower_bound = np.array([110, 100, 30])
    upper_bound = np.array([130, 255, 255])
    
    img_threshed = cv2.inRange(img_hsv, lower_bound, upper_bound)
    return img_threshed

def process_headless(img):
    """
    Process image in headless mode and print results
    """
    img_threshed = get_thresholded_image(img)
    
    # Calculate moments
    moments = cv2.moments(img_threshed, False)
    
    if moments['m00'] != 0:
        x = int(moments['m10'] / moments['m00'])
        y = int(moments['m01'] / moments['m00'])
        print(f"Yellow object detected at position: ({x}, {y})")
        print(f"Object area: {moments['m00']}")
    else:
        print("No yellow object detected")
    
    # Save the processed image
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/headless_result.jpg", img_threshed)
    print("Processed image saved to outputs/headless_result.jpg")

def main():
    # Check for headless mode
    headless = len(sys.argv) > 1 and sys.argv[1] == "--headless"
    
    # Initialize camera capture
    capture = cv2.VideoCapture(0)
    
    if not capture.isOpened():
        print("Could not initialize camera")
        if headless:
            print("Running in headless mode - using test image instead")
            # Create a test image for headless mode
            test_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(test_img, (100, 100), (200, 200), (0, 255, 255), -1)  # Yellow rectangle
            process_headless(test_img)
            return 0
        return -1
    
    if not headless:
        # Create windows only in GUI mode
        cv2.namedWindow("thresh")
        cv2.namedWindow("video")
        cv2.namedWindow("scribble")
    else:
        print("Running in headless mode - processing frames without GUI")
    
    img_scribble = None
    pos_x, pos_y = 0, 0
    
    frame_count = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            print("Error reading frame")
            break
        
        # Initialize scribble image on first frame
        if img_scribble is None:
            img_scribble = np.zeros_like(frame)
        
        # Get thresholded image
        img_yellow_thresh = get_thresholded_image(frame)
        
        # Calculate moments for object tracking
        moments = cv2.moments(img_yellow_thresh)
        
        if moments['m00'] != 0:  # Avoid division by zero
            last_x, last_y = pos_x, pos_y
            
            # Calculate centroid position
            pos_x = int(moments['m10'] / moments['m00'])
            pos_y = int(moments['m01'] / moments['m00'])
            
            print(f"Position ({pos_x}, {pos_y})")
            
            # Draw line connecting previous and current positions
            if last_x > 0 and last_y > 0 and pos_x > 0 and pos_y > 0:
                cv2.line(img_scribble, (pos_x, pos_y), (last_x, last_y), (0, 255, 255), 5)
        
        if headless:
            # In headless mode, process a few frames and exit
            frame_count += 1
            if frame_count >= 10:
                print("Headless mode: processed 10 frames, saving results...")
                os.makedirs("outputs", exist_ok=True)
                cv2.imwrite("outputs/headless_thresh.jpg", img_yellow_thresh)
                cv2.imwrite("outputs/headless_scribble.jpg", img_scribble)
                print("Results saved to outputs/")
                break
        else:
            # Add scribble overlay to frame
            frame_with_scribble = cv2.add(frame, img_scribble)
            
            # Display images
            cv2.imshow("thresh", img_yellow_thresh)
            cv2.imshow("video", frame_with_scribble)
            cv2.imshow("scribble", img_scribble)
            
            # Check for ESC key press
            key = cv2.waitKey(33) & 0xFF
            if key == 27:  # ESC key
                break
    
    # Cleanup
    capture.release()
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    main()