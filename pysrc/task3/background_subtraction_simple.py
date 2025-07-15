import cv2
import numpy as np

def main():
    """
    Simple background subtraction using grayscale analysis
    Equivalent to change.cpp
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not initialize camera")
        return -1
    
    # Create windows
    cv2.namedWindow("Original Frame")
    cv2.namedWindow("Grayscale")
    cv2.namedWindow("Background Model")
    cv2.namedWindow("Absolute Difference")
    cv2.namedWindow("Subtraction")
    cv2.namedWindow("Thresholded")
    cv2.namedWindow("Morphological")
    cv2.namedWindow("Foreground Objects")
    
    print("Simple background subtraction using grayscale analysis")
    print("Building background model from first 30 frames...")
    
    # Initialize variables
    background = None
    frame_count = 0
    background_frames = 30  # Number of frames to build background model
    threshold_value = 20    # Threshold for motion detection (equivalent to C++ code)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Convert current frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Build background model during initial frames
        if frame_count <= background_frames:
            print(f"Building background model... Frame {frame_count}/{background_frames}")
            
            if background is None:
                # Initialize background with first frame
                background = gray_frame.astype(np.float32)
            else:
                # Update background model using running average
                alpha = 0.1  # Learning rate
                background = (1 - alpha) * background + alpha * gray_frame.astype(np.float32)
            
            # Display current frame during background learning
            cv2.imshow("Original Frame", frame)
            cv2.imshow("Grayscale", gray_frame)
            
            if frame_count == background_frames:
                print("Background model established. Starting motion detection...")
                # Convert background to uint8 for processing
                background_uint8 = background.astype(np.uint8)
                cv2.imshow("Background Model", background_uint8)
            
        else:
            # Perform background subtraction
            background_uint8 = background.astype(np.uint8)
            
            # Calculate absolute difference (equivalent to cvAbsDiff)
            abs_diff = cv2.absdiff(gray_frame, background_uint8)
            
            # Calculate regular subtraction (equivalent to cvSub)
            sub_diff = cv2.subtract(gray_frame, background_uint8)
            
            # Apply threshold to create binary mask (threshold = 20)
            _, thresholded = cv2.threshold(abs_diff, threshold_value, 255, cv2.THRESH_BINARY)
            
            # Apply morphological dilation to reduce noise (equivalent to cvDilate)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            morphological = cv2.dilate(thresholded, kernel, iterations=1)
            
            # Optional: Add erosion for better noise reduction
            morphological = cv2.erode(morphological, kernel, iterations=1)
            
            # Extract foreground objects using mask (equivalent to cvCopy with mask)
            foreground = np.zeros_like(frame)
            foreground[morphological == 255] = frame[morphological == 255]
            
            # Calculate motion statistics
            motion_pixels = cv2.countNonZero(morphological)
            total_pixels = morphological.shape[0] * morphological.shape[1]
            motion_percentage = (motion_pixels / total_pixels) * 100
            
            # Add text overlays
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Motion: {motion_percentage:.2f}%", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Threshold: {threshold_value}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(foreground, "Detected Objects", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display all processing stages
            cv2.imshow("Original Frame", frame)
            cv2.imshow("Grayscale", gray_frame)
            cv2.imshow("Background Model", background_uint8)
            cv2.imshow("Absolute Difference", abs_diff)
            cv2.imshow("Subtraction", sub_diff)
            cv2.imshow("Thresholded", thresholded)
            cv2.imshow("Morphological", morphological)
            cv2.imshow("Foreground Objects", foreground)
            
            # Print motion information for significant motion
            if motion_percentage > 1.0:
                print(f"Frame {frame_count}: Motion detected - {motion_percentage:.2f}% of image")
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):  # Reset background
            background = None
            frame_count = 0
            print("Background model reset")
        elif key == ord('t'):  # Adjust threshold
            print(f"Current threshold: {threshold_value}")
            try:
                new_threshold = int(input("Enter new threshold (0-255): "))
                threshold_value = max(0, min(255, new_threshold))
                print(f"Threshold set to: {threshold_value}")
            except ValueError:
                print("Invalid threshold value")
        elif key == ord('s'):  # Save current frame and results
            if frame_count > background_frames:
                cv2.imwrite(f"simple_bg_original_{frame_count}.jpg", frame)
                cv2.imwrite(f"simple_bg_background_{frame_count}.jpg", background_uint8)
                cv2.imwrite(f"simple_bg_foreground_{frame_count}.jpg", foreground)
                cv2.imwrite(f"simple_bg_threshold_{frame_count}.jpg", thresholded)
                cv2.imwrite(f"simple_bg_morphological_{frame_count}.jpg", morphological)
                print(f"Frame {frame_count} and analysis results saved")
        elif key == ord('u'):  # Update background with current frame
            if frame_count > background_frames:
                alpha = 0.1
                background = (1 - alpha) * background + alpha * gray_frame.astype(np.float32)
                print("Background model updated with current frame")
        elif key == ord('f'):  # Freeze background (stop adaptation)
            print("Background model frozen")
        elif key == ord('h'):  # Show help
            print("\nKey Controls:")
            print("ESC - Exit")
            print("r - Reset background model")
            print("t - Adjust threshold")
            print("s - Save current frame and results")
            print("u - Update background with current frame")
            print("f - Freeze background model")
            print("h - Show this help")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total frames processed: {frame_count}")

def process_video_file(video_path):
    """
    Process a video file instead of camera input
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Could not open video file: {video_path}")
        return -1
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {video_path}")
    print(f"FPS: {fps}, Total frames: {total_frames}")
    
    # Similar processing as main() but optimized for video files
    background = None
    frame_count = 0
    background_frames = min(30, total_frames // 10)  # Adapt to video length
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if frame_count <= background_frames:
            if background is None:
                background = gray_frame.astype(np.float32)
            else:
                alpha = 0.1
                background = (1 - alpha) * background + alpha * gray_frame.astype(np.float32)
        else:
            background_uint8 = background.astype(np.uint8)
            abs_diff = cv2.absdiff(gray_frame, background_uint8)
            _, thresholded = cv2.threshold(abs_diff, 20, 255, cv2.THRESH_BINARY)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            morphological = cv2.dilate(thresholded, kernel, iterations=1)
            morphological = cv2.erode(morphological, kernel, iterations=1)
            
            foreground = np.zeros_like(frame)
            foreground[morphological == 255] = frame[morphological == 255]
            
            # Display results
            cv2.imshow("Video Frame", frame)
            cv2.imshow("Motion Detection", morphological)
            cv2.imshow("Foreground", foreground)
        
        # Control playback speed
        if cv2.waitKey(int(1000/fps)) & 0xFF == 27:
            break
        
        print(f"Processing frame {frame_count}/{total_frames}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Video file mode
        process_video_file(sys.argv[1])
    else:
        # Real-time camera mode
        main()