import cv2
import numpy as np

def main():
    """
    Motion detection through frame differencing
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not initialize camera")
        return -1
    
    cv2.namedWindow("Current Frame")
    cv2.namedWindow("Previous Frame")
    cv2.namedWindow("Frame Difference")
    cv2.namedWindow("Absolute Difference")
    cv2.namedWindow("Motion Detection")
    
    # Initialize variables
    prev_frame = None
    frame_count = 0
    motion_threshold = 30
    
    print("Motion detection started. Press ESC to exit.")
    print("Press 't' to adjust threshold, 'r' to reset background.")
    
    while True:
        ret, current_frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Convert to grayscale for motion detection
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is not None:
            # Calculate frame difference (subtraction)
            diff_frame = cv2.subtract(gray_current, prev_frame)
            
            # Calculate absolute difference
            abs_diff = cv2.absdiff(gray_current, prev_frame)
            
            # Apply threshold to get binary motion mask
            _, motion_mask = cv2.threshold(abs_diff, motion_threshold, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations to reduce noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours for motion regions
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw motion regions on current frame
            motion_display = current_frame.copy()
            motion_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Filter small motion areas
                    motion_area += area
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(motion_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.drawContours(motion_display, [contour], -1, (0, 0, 255), 2)
            
            # Calculate motion percentage
            total_pixels = gray_current.shape[0] * gray_current.shape[1]
            motion_percentage = (motion_area / total_pixels) * 100
            
            # Add text overlay
            cv2.putText(motion_display, f"Motion: {motion_percentage:.2f}%", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(motion_display, f"Threshold: {motion_threshold}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(motion_display, f"Frame: {frame_count}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Convert single-channel images to 3-channel for display
            diff_display = cv2.merge([diff_frame, diff_frame, diff_frame])
            abs_diff_display = cv2.merge([abs_diff, abs_diff, abs_diff])
            motion_mask_display = cv2.merge([motion_mask, motion_mask, motion_mask])
            prev_frame_display = cv2.merge([prev_frame, prev_frame, prev_frame])
            
            # Display all images
            cv2.imshow("Current Frame", current_frame)
            cv2.imshow("Previous Frame", prev_frame_display)
            cv2.imshow("Frame Difference", diff_display)
            cv2.imshow("Absolute Difference", abs_diff_display)
            cv2.imshow("Motion Detection", motion_display)
            
            # Print motion information
            if motion_percentage > 5.0:  # Significant motion detected
                print(f"Frame {frame_count}: Motion detected - {motion_percentage:.2f}% of image")
        else:
            # First frame - just display current frame
            cv2.imshow("Current Frame", current_frame)
            print("Initializing motion detection...")
        
        # Store current frame as previous for next iteration
        prev_frame = gray_current.copy()
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == ord('t'):  # Adjust threshold
            print(f"Current threshold: {motion_threshold}")
            new_threshold = input("Enter new threshold (0-255): ")
            try:
                motion_threshold = max(0, min(255, int(new_threshold)))
                print(f"Threshold set to: {motion_threshold}")
            except ValueError:
                print("Invalid threshold value")
        elif key == ord('r'):  # Reset background
            prev_frame = None
            frame_count = 0
            print("Background reset")
        elif key == ord('s'):  # Save current frame
            if 'motion_display' in locals():
                cv2.imwrite(f"motion_frame_{frame_count}.jpg", motion_display)
                print(f"Frame saved as motion_frame_{frame_count}.jpg")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()