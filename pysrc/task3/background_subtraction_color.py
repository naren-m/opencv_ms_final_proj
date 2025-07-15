import cv2
import numpy as np

def main():
    """
    Advanced background subtraction using color channel analysis (BGR channels)
    Equivalent to backChange.cpp
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not initialize camera")
        return -1
    
    # Create windows
    cv2.namedWindow("Original Frame")
    cv2.namedWindow("Background Model")
    cv2.namedWindow("Blue Difference")
    cv2.namedWindow("Green Difference") 
    cv2.namedWindow("Red Difference")
    cv2.namedWindow("Combined Motion")
    cv2.namedWindow("Foreground Objects")
    
    print("Background subtraction with color channel analysis")
    print("Building background model from first 30 frames...")
    
    # Initialize variables
    background_b = None
    background_g = None
    background_r = None
    frame_count = 0
    background_frames = 30  # Number of frames to build background model
    
    # Thresholds for each channel (can be adjusted)
    threshold_b = 20
    threshold_g = 20
    threshold_r = 20
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Split frame into BGR channels
        b, g, r = cv2.split(frame)
        
        # Build background model during initial frames
        if frame_count <= background_frames:
            print(f"Building background model... Frame {frame_count}/{background_frames}")
            
            if background_b is None:
                # Initialize background with first frame
                background_b = b.astype(np.float32)
                background_g = g.astype(np.float32)
                background_r = r.astype(np.float32)
            else:
                # Update background model using running average
                alpha = 0.1  # Learning rate
                background_b = (1 - alpha) * background_b + alpha * b.astype(np.float32)
                background_g = (1 - alpha) * background_g + alpha * g.astype(np.float32)
                background_r = (1 - alpha) * background_r + alpha * r.astype(np.float32)
            
            # Display current frame during background learning
            cv2.imshow("Original Frame", frame)
            
            if frame_count == background_frames:
                print("Background model established. Starting motion detection...")
            
        else:
            # Perform background subtraction on each channel
            
            # Calculate absolute differences
            diff_b = cv2.absdiff(b, background_b.astype(np.uint8))
            diff_g = cv2.absdiff(g, background_g.astype(np.uint8))
            diff_r = cv2.absdiff(r, background_r.astype(np.uint8))
            
            # Apply thresholds to create binary masks
            _, mask_b = cv2.threshold(diff_b, threshold_b, 255, cv2.THRESH_BINARY)
            _, mask_g = cv2.threshold(diff_g, threshold_g, 255, cv2.THRESH_BINARY)
            _, mask_r = cv2.threshold(diff_r, threshold_r, 255, cv2.THRESH_BINARY)
            
            # Combine masks (OR operation)
            combined_mask = cv2.bitwise_or(mask_b, mask_g)
            combined_mask = cv2.bitwise_or(combined_mask, mask_r)
            
            # Optional: Apply morphological operations to reduce noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            
            # Extract foreground objects
            foreground = np.zeros_like(frame)
            foreground[combined_mask == 255] = frame[combined_mask == 255]
            
            # Create background display image
            background_display = cv2.merge([
                background_b.astype(np.uint8),
                background_g.astype(np.uint8), 
                background_r.astype(np.uint8)
            ])
            
            # Create colored difference displays
            diff_b_colored = cv2.merge([diff_b, np.zeros_like(diff_b), np.zeros_like(diff_b)])
            diff_g_colored = cv2.merge([np.zeros_like(diff_g), diff_g, np.zeros_like(diff_g)])
            diff_r_colored = cv2.merge([np.zeros_like(diff_r), np.zeros_like(diff_r), diff_r])
            
            # Create combined motion visualization
            motion_display = cv2.merge([mask_r, mask_g, mask_b])
            
            # Add text overlays
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(foreground, "Detected Motion", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Calculate motion statistics
            motion_pixels = cv2.countNonZero(combined_mask)
            total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
            motion_percentage = (motion_pixels / total_pixels) * 100
            
            cv2.putText(frame, f"Motion: {motion_percentage:.2f}%", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display all images
            cv2.imshow("Original Frame", frame)
            cv2.imshow("Background Model", background_display)
            cv2.imshow("Blue Difference", diff_b_colored)
            cv2.imshow("Green Difference", diff_g_colored)
            cv2.imshow("Red Difference", diff_r_colored)
            cv2.imshow("Combined Motion", motion_display)
            cv2.imshow("Foreground Objects", foreground)
            
            # Print motion information for significant motion
            if motion_percentage > 1.0:
                print(f"Frame {frame_count}: Motion detected - {motion_percentage:.2f}% of image")
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):  # Reset background
            background_b = None
            background_g = None
            background_r = None
            frame_count = 0
            print("Background model reset")
        elif key == ord('t'):  # Adjust thresholds
            print(f"Current thresholds - B:{threshold_b}, G:{threshold_g}, R:{threshold_r}")
            try:
                new_threshold = int(input("Enter new threshold (0-255): "))
                threshold_b = threshold_g = threshold_r = max(0, min(255, new_threshold))
                print(f"Thresholds set to: {threshold_b}")
            except ValueError:
                print("Invalid threshold value")
        elif key == ord('s'):  # Save current frame and results
            if frame_count > background_frames:
                cv2.imwrite(f"original_frame_{frame_count}.jpg", frame)
                cv2.imwrite(f"background_model_{frame_count}.jpg", background_display)
                cv2.imwrite(f"foreground_objects_{frame_count}.jpg", foreground)
                cv2.imwrite(f"motion_mask_{frame_count}.jpg", combined_mask)
                print(f"Frame {frame_count} and analysis results saved")
        elif key == ord('u'):  # Update background with current frame
            if frame_count > background_frames:
                alpha = 0.1
                background_b = (1 - alpha) * background_b + alpha * b.astype(np.float32)
                background_g = (1 - alpha) * background_g + alpha * g.astype(np.float32)
                background_r = (1 - alpha) * background_r + alpha * r.astype(np.float32)
                print("Background model updated with current frame")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total frames processed: {frame_count}")

if __name__ == "__main__":
    main()