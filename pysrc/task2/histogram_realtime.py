import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_histogram_image(hist, color, width=400, height=300):
    """
    Create a visual representation of histogram data (equivalent to imHist function)
    """
    # Create blank image
    hist_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Normalize histogram
    if hist.max() > 0:
        normalized_hist = hist * (height - 50) / hist.max()
    else:
        normalized_hist = hist
    
    # Calculate bin width
    bin_width = width // len(hist)
    
    # Draw histogram bars
    for i in range(len(hist)):
        x1 = i * bin_width
        x2 = (i + 1) * bin_width
        y1 = height - 10
        y2 = height - int(normalized_hist[i]) - 10
        
        # Create polygon points for filled bar
        pts = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]], np.int32)
        
        # Fill the bar with specified color
        cv2.fillPoly(hist_img, [pts], color)
    
    return hist_img

def create_combined_histogram(hist_b, hist_g, hist_r, width=400, height=300):
    """
    Create a combined histogram showing all three color channels
    """
    hist_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Find maximum value across all histograms for normalization
    max_val = max(hist_b.max(), hist_g.max(), hist_r.max())
    
    if max_val > 0:
        norm_b = hist_b * (height - 50) / max_val
        norm_g = hist_g * (height - 50) / max_val
        norm_r = hist_r * (height - 50) / max_val
    else:
        norm_b = norm_g = norm_r = np.zeros_like(hist_b)
    
    bin_width = width // len(hist_b)
    
    for i in range(len(hist_b)):
        x = i * bin_width
        
        # Draw lines for each color channel
        if norm_b[i] > 0:
            cv2.line(hist_img, (x, height-10), (x, height-int(norm_b[i])-10), (255, 0, 0), 2)
        if norm_g[i] > 0:
            cv2.line(hist_img, (x, height-10), (x, height-int(norm_g[i])-10), (0, 255, 0), 2)
        if norm_r[i] > 0:
            cv2.line(hist_img, (x, height-10), (x, height-int(norm_r[i])-10), (0, 0, 255), 2)
    
    # Add legend
    cv2.putText(hist_img, "B", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(hist_img, "G", (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(hist_img, "R", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return hist_img

def main():
    """
    Real-time histogram visualization from webcam (equivalent to original C++ implementation)
    """
    # Initialize camera capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not initialize camera")
        return -1
    
    # Create windows
    cv2.namedWindow("Webcam Feed")
    cv2.namedWindow("Blue Histogram")
    cv2.namedWindow("Green Histogram")
    cv2.namedWindow("Red Histogram")
    cv2.namedWindow("Combined Histogram")
    
    print("Real-time histogram visualization started.")
    print("Press 'ESC' to exit, 's' to save current frame and histograms.")
    
    frame_count = 0
    
    while True:
        # Capture frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break
        
        frame_count += 1
        
        # Split BGR channels
        b, g, r = cv2.split(frame)
        
        # Calculate histograms for each channel
        hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
        hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
        
        # Flatten histograms (calcHist returns 2D array)
        hist_b = hist_b.flatten()
        hist_g = hist_g.flatten()
        hist_r = hist_r.flatten()
        
        # Create histogram visualizations
        hist_img_b = create_histogram_image(hist_b, (255, 0, 0))  # Blue
        hist_img_g = create_histogram_image(hist_g, (0, 255, 0))  # Green
        hist_img_r = create_histogram_image(hist_r, (0, 0, 255))  # Red
        hist_img_combined = create_combined_histogram(hist_b, hist_g, hist_r)
        
        # Add frame counter to webcam feed
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display all images
        cv2.imshow("Webcam Feed", frame)
        cv2.imshow("Blue Histogram", hist_img_b)
        cv2.imshow("Green Histogram", hist_img_g)
        cv2.imshow("Red Histogram", hist_img_r)
        cv2.imshow("Combined Histogram", hist_img_combined)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == ord('s'):  # Save current frame and histograms
            cv2.imwrite(f"frame_{frame_count}.jpg", frame)
            cv2.imwrite(f"hist_blue_{frame_count}.jpg", hist_img_b)
            cv2.imwrite(f"hist_green_{frame_count}.jpg", hist_img_g)
            cv2.imwrite(f"hist_red_{frame_count}.jpg", hist_img_r)
            cv2.imwrite(f"hist_combined_{frame_count}.jpg", hist_img_combined)
            print(f"Frame {frame_count} and histograms saved")
        elif key == ord('p'):  # Print histogram statistics
            print(f"\nFrame {frame_count} Histogram Statistics:")
            print(f"Blue   - Min: {hist_b.min():.0f}, Max: {hist_b.max():.0f}, Mean: {hist_b.mean():.0f}")
            print(f"Green  - Min: {hist_g.min():.0f}, Max: {hist_g.max():.0f}, Mean: {hist_g.mean():.0f}")
            print(f"Red    - Min: {hist_r.min():.0f}, Max: {hist_r.max():.0f}, Mean: {hist_r.mean():.0f}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total frames processed: {frame_count}")

def plot_static_histogram(image_path=None):
    """
    Alternative function to plot histograms using matplotlib (for static images)
    """
    if image_path:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return
    else:
        # Capture single frame from camera
        cap = cv2.VideoCapture(0)
        ret, img = cap.read()
        cap.release()
        if not ret:
            print("Could not capture from camera")
            return
    
    # Split channels
    b, g, r = cv2.split(img)
    
    # Calculate histograms
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
    
    # Plot using matplotlib
    plt.figure(figsize=(12, 8))
    
    # Original image
    plt.subplot(2, 2, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    # Individual histograms
    plt.subplot(2, 2, 2)
    plt.plot(hist_b, color='blue', label='Blue', alpha=0.7)
    plt.title('Blue Channel Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(hist_g, color='green', label='Green', alpha=0.7)
    plt.title('Green Channel Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(hist_r, color='red', label='Red', alpha=0.7)
    plt.title('Red Channel Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "static":
        # Static histogram mode with matplotlib
        image_path = sys.argv[2] if len(sys.argv) > 2 else None
        plot_static_histogram(image_path)
    else:
        # Real-time histogram mode (default)
        main()