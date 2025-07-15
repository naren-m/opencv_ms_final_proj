import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_histogram(image):
    """
    Create and display 2D histogram for H-S channels
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate 2D histogram for H-S channels
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    
    # Normalize histogram
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    
    return hist

def main():
    """
    Histogram comparison between consecutive video frames
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not initialize camera")
        return -1
    
    cv2.namedWindow("Video")
    cv2.namedWindow("Histogram")
    
    prev_hist = None
    correlation_values = []
    frame_count = 0
    
    print("Starting histogram comparison. Press ESC to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate current frame histogram
        current_hist = draw_histogram(frame)
        
        # Compare with previous histogram if available
        if prev_hist is not None:
            # Calculate correlation
            correlation = cv2.compareHist(prev_hist, current_hist, cv2.HISTCMP_CORREL)
            correlation_values.append(correlation)
            
            print(f"Frame {frame_count}: Correlation = {correlation:.4f}")
            
            # Display correlation on frame
            cv2.putText(frame, f"Correlation: {correlation:.4f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Create histogram visualization
        hist_img = np.zeros((400, 512, 3), dtype=np.uint8)
        
        # Draw 2D histogram as image
        hist_display = cv2.resize(current_hist, (512, 400))
        hist_display = cv2.normalize(hist_display, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        hist_img[:, :, 0] = hist_display
        hist_img[:, :, 1] = hist_display
        hist_img[:, :, 2] = hist_display
        
        # Display images
        cv2.imshow("Video", frame)
        cv2.imshow("Histogram", hist_img)
        
        # Store current histogram for next comparison
        prev_hist = current_hist.copy()
        frame_count += 1
        
        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Plot correlation values over time
    if correlation_values:
        plt.figure(figsize=(10, 6))
        plt.plot(correlation_values)
        plt.title('Histogram Correlation Between Consecutive Frames')
        plt.xlabel('Frame Number')
        plt.ylabel('Correlation Value')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()