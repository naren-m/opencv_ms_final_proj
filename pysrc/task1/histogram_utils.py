import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_histogram_2d(image, h_bins=50, s_bins=60):
    """
    Create 2D histogram for H-S channels and return visualization
    Equivalent to the drawHistogram function in functions.h
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Split channels (we only need H and S for 2D histogram)
    h, s, v = cv2.split(hsv)
    
    # Calculate 2D histogram for H-S channels
    hist = cv2.calcHist([hsv], [0, 1], None, [h_bins, s_bins], [0, 180, 0, 256])
    
    # Normalize histogram
    hist_normalized = cv2.normalize(hist, None, 0, 255, cv2.NORM_MINMAX)
    
    return hist, hist_normalized

def visualize_histogram_2d(hist, title="2D Histogram (H-S)"):
    """
    Visualize 2D histogram using matplotlib
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(hist.T, cmap='hot', interpolation='nearest', origin='lower')
    plt.colorbar(label='Frequency')
    plt.xlabel('Hue')
    plt.ylabel('Saturation')
    plt.title(title)
    plt.show()

def draw_histogram_opencv(hist, width=512, height=400):
    """
    Draw histogram visualization using OpenCV (similar to original C++ implementation)
    """
    # Create blank image for histogram
    hist_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Resize histogram to fit display
    hist_resized = cv2.resize(hist, (width, height))
    
    # Normalize to 0-255 range
    hist_display = cv2.normalize(hist_resized, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Convert to 3-channel for display
    hist_img[:, :, 0] = hist_display
    hist_img[:, :, 1] = hist_display  
    hist_img[:, :, 2] = hist_display
    
    return hist_img

def create_1d_histograms(image):
    """
    Create individual 1D histograms for H, S, V channels
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Calculate histograms
    hist_h = cv2.calcHist([h], [0], None, [180], [0, 180])
    hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
    hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])
    
    return hist_h, hist_s, hist_v

def plot_1d_histograms(hist_h, hist_s, hist_v):
    """
    Plot 1D histograms using matplotlib
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Hue histogram
    ax1.plot(hist_h, color='red')
    ax1.set_title('Hue Histogram')
    ax1.set_xlabel('Hue Value')
    ax1.set_ylabel('Frequency')
    ax1.set_xlim([0, 180])
    
    # Saturation histogram
    ax2.plot(hist_s, color='green')
    ax2.set_title('Saturation Histogram')
    ax2.set_xlabel('Saturation Value')
    ax2.set_ylabel('Frequency')
    ax2.set_xlim([0, 256])
    
    # Value histogram
    ax3.plot(hist_v, color='blue')
    ax3.set_title('Value Histogram')
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Frequency')
    ax3.set_xlim([0, 256])
    
    plt.tight_layout()
    plt.show()

def compare_histograms(hist1, hist2, method=cv2.HISTCMP_CORREL):
    """
    Compare two histograms using specified method
    """
    methods = {
        cv2.HISTCMP_CORREL: "Correlation",
        cv2.HISTCMP_CHISQR: "Chi-Square",
        cv2.HISTCMP_INTERSECT: "Intersection",
        cv2.HISTCMP_BHATTACHARYYA: "Bhattacharyya"
    }
    
    comparison = cv2.compareHist(hist1, hist2, method)
    method_name = methods.get(method, "Unknown")
    
    return comparison, method_name

# Example usage and test function
def test_histogram_functions():
    """
    Test function to demonstrate histogram utilities
    """
    # Try to load an image or use camera
    cap = cv2.VideoCapture(0)
    ret, image = cap.read()
    cap.release()
    
    if not ret:
        print("Could not capture from camera")
        return
    
    # Create 2D histogram
    hist_2d, hist_normalized = draw_histogram_2d(image)
    
    # Visualize using OpenCV
    hist_img = draw_histogram_opencv(hist_normalized)
    cv2.imshow("2D Histogram", hist_img)
    cv2.imshow("Original", image)
    
    # Create 1D histograms
    hist_h, hist_s, hist_v = create_1d_histograms(image)
    
    print("Press any key to close windows and show matplotlib plots...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Show matplotlib visualizations
    visualize_histogram_2d(hist_2d)
    plot_1d_histograms(hist_h, hist_s, hist_v)

if __name__ == "__main__":
    test_histogram_functions()