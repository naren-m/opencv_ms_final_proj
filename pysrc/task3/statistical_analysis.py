import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_pixel_statistics_array(num_frames=60):
    """
    Calculate standard deviation of pixel differences using array-based approach
    Equivalent to stdDev.cpp
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not initialize camera")
        return -1
    
    print(f"Collecting {num_frames} frames for statistical analysis...")
    
    # Get frame dimensions
    ret, first_frame = cap.read()
    if not ret:
        print("Could not read first frame")
        return -1
    
    gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    height, width = gray_frame.shape
    
    # Storage for pixel differences
    pixel_diffs = []
    frames = []
    
    prev_frame = gray_frame.copy()
    frames.append(prev_frame.copy())
    
    cv2.namedWindow("Current Frame")
    cv2.namedWindow("Frame Difference")
    
    for frame_num in range(1, num_frames):
        ret, current_frame = cap.read()
        if not ret:
            break
        
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        frames.append(current_gray.copy())
        
        # Calculate frame difference
        diff_frame = cv2.absdiff(current_gray, prev_frame)
        pixel_diffs.append(diff_frame.flatten())  # Flatten to 1D array
        
        # Display current processing
        cv2.putText(current_frame, f"Frame: {frame_num}/{num_frames-1}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(current_frame, "Collecting data...", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Current Frame", current_frame)
        cv2.imshow("Frame Difference", diff_frame)
        
        prev_frame = current_gray.copy()
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop early
            break
        
        print(f"Processed frame {frame_num}/{num_frames-1}")
    
    cap.release()
    
    # Convert to numpy array for statistical calculations
    pixel_diffs_array = np.array(pixel_diffs)
    
    print("\nCalculating statistics...")
    
    # Calculate statistics for each pixel position across all frames
    mean_diffs = np.mean(pixel_diffs_array, axis=0)
    mean_squared_diffs = np.mean(pixel_diffs_array**2, axis=0)
    
    # Calculate variance: E(X²) - (E(X))²
    variance_diffs = mean_squared_diffs - mean_diffs**2
    std_dev_diffs = np.sqrt(variance_diffs)
    
    # Reshape back to image dimensions
    mean_image = mean_diffs.reshape(height, width)
    variance_image = variance_diffs.reshape(height, width)
    std_dev_image = std_dev_diffs.reshape(height, width)
    
    # Normalize for display (0-255)
    mean_display = cv2.normalize(mean_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    variance_display = cv2.normalize(variance_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    std_dev_display = cv2.normalize(std_dev_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Display results
    cv2.namedWindow("Mean Differences")
    cv2.namedWindow("Variance")
    cv2.namedWindow("Standard Deviation")
    
    cv2.imshow("Mean Differences", mean_display)
    cv2.imshow("Variance", variance_display)
    cv2.imshow("Standard Deviation", std_dev_display)
    
    # Print overall statistics
    print(f"\nStatistical Analysis Results ({len(pixel_diffs)} frames):")
    print(f"Overall mean difference: {np.mean(mean_diffs):.4f}")
    print(f"Overall variance: {np.mean(variance_diffs):.4f}")
    print(f"Overall standard deviation: {np.mean(std_dev_diffs):.4f}")
    print(f"Min std dev: {np.min(std_dev_diffs):.4f}")
    print(f"Max std dev: {np.max(std_dev_diffs):.4f}")
    
    # Find most and least variable regions
    max_var_idx = np.unravel_index(np.argmax(variance_image), variance_image.shape)
    min_var_idx = np.unravel_index(np.argmin(variance_image), variance_image.shape)
    
    print(f"Most variable pixel at {max_var_idx}: variance = {variance_image[max_var_idx]:.4f}")
    print(f"Least variable pixel at {min_var_idx}: variance = {variance_image[min_var_idx]:.4f}")
    
    print("\nPress any key to show matplotlib plots, ESC to exit")
    key = cv2.waitKey(0)
    
    if key != 27:  # If not ESC, show matplotlib plots
        plot_statistical_analysis(mean_image, variance_image, std_dev_image, pixel_diffs_array)
    
    cv2.destroyAllWindows()
    
    return {
        'mean_image': mean_image,
        'variance_image': variance_image,
        'std_dev_image': std_dev_image,
        'pixel_diffs': pixel_diffs_array,
        'frames_processed': len(pixel_diffs)
    }

def calculate_pixel_statistics_opencv(num_frames=60):
    """
    Calculate standard deviation using OpenCV image structures
    Equivalent to stdMan.cpp
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not initialize camera")
        return -1
    
    print(f"Collecting {num_frames} frames for OpenCV-based statistical analysis...")
    
    # Get frame dimensions
    ret, first_frame = cap.read()
    if not ret:
        print("Could not read first frame")
        return -1
    
    gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    height, width = gray_frame.shape
    
    # Initialize accumulator images (using double precision)
    sum_image = np.zeros((height, width), dtype=np.float64)
    sum_squared_image = np.zeros((height, width), dtype=np.float64)
    
    prev_frame = gray_frame.astype(np.float64)
    frame_count = 0
    
    cv2.namedWindow("Processing")
    
    for frame_num in range(1, num_frames):
        ret, current_frame = cap.read()
        if not ret:
            break
        
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        
        # Calculate frame difference (equivalent to cvSub)
        diff_image = np.abs(current_gray - prev_frame)
        
        # Accumulate statistics
        sum_image += diff_image
        sum_squared_image += diff_image ** 2
        
        frame_count += 1
        
        # Display progress
        display_frame = current_frame.copy()
        cv2.putText(display_frame, f"Frame: {frame_num}/{num_frames-1}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "OpenCV Statistical Analysis", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Processing", display_frame)
        
        prev_frame = current_gray.copy()
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
        print(f"Processed frame {frame_num}/{num_frames-1}")
    
    cap.release()
    
    # Calculate final statistics
    if frame_count > 0:
        mean_image = sum_image / frame_count
        mean_squared_image = sum_squared_image / frame_count
        
        # Variance = E(X²) - (E(X))²
        variance_image = mean_squared_image - (mean_image ** 2)
        std_dev_image = np.sqrt(np.maximum(variance_image, 0))  # Ensure non-negative
        
        # Convert to uint8 for display
        mean_display = cv2.normalize(mean_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        variance_display = cv2.normalize(variance_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        std_dev_display = cv2.normalize(std_dev_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Display results
        cv2.imshow("Mean (OpenCV)", mean_display)
        cv2.imshow("Variance (OpenCV)", variance_display)  
        cv2.imshow("Standard Deviation (OpenCV)", std_dev_display)
        
        # Print statistics
        print(f"\nOpenCV Statistical Analysis Results ({frame_count} frames):")
        print(f"Mean difference: {np.mean(mean_image):.6f}")
        print(f"Variance: {np.mean(variance_image):.6f}")
        print(f"Standard deviation: {np.mean(std_dev_image):.6f}")
        
        # Print some specific pixel values (equivalent to PrintImage function)
        print(f"\nSample pixel values at center ({height//2}, {width//2}):")
        center_y, center_x = height//2, width//2
        print(f"Mean: {mean_image[center_y, center_x]:.6f}")
        print(f"Variance: {variance_image[center_y, center_x]:.6f}")
        print(f"Std Dev: {std_dev_image[center_y, center_x]:.6f}")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return {
            'mean_image': mean_image,
            'variance_image': variance_image,
            'std_dev_image': std_dev_image,
            'frames_processed': frame_count
        }
    
    cv2.destroyAllWindows()
    return None

def plot_statistical_analysis(mean_image, variance_image, std_dev_image, pixel_diffs_array):
    """
    Create matplotlib visualizations of statistical analysis
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Mean differences
    im1 = axes[0, 0].imshow(mean_image, cmap='gray')
    axes[0, 0].set_title('Mean Differences')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Variance
    im2 = axes[0, 1].imshow(variance_image, cmap='hot')
    axes[0, 1].set_title('Variance')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Standard deviation
    im3 = axes[0, 2].imshow(std_dev_image, cmap='viridis')
    axes[0, 2].set_title('Standard Deviation')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Histogram of mean differences
    axes[1, 0].hist(mean_image.flatten(), bins=50, alpha=0.7)
    axes[1, 0].set_title('Distribution of Mean Differences')
    axes[1, 0].set_xlabel('Mean Difference Value')
    axes[1, 0].set_ylabel('Frequency')
    
    # Histogram of variances
    axes[1, 1].hist(variance_image.flatten(), bins=50, alpha=0.7, color='orange')
    axes[1, 1].set_title('Distribution of Variances')
    axes[1, 1].set_xlabel('Variance Value')
    axes[1, 1].set_ylabel('Frequency')
    
    # Time series of overall frame differences
    overall_diffs = np.mean(pixel_diffs_array, axis=1)
    axes[1, 2].plot(overall_diffs)
    axes[1, 2].set_title('Overall Frame Differences Over Time')
    axes[1, 2].set_xlabel('Frame Number')
    axes[1, 2].set_ylabel('Mean Difference')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to choose between array-based and OpenCV-based analysis
    """
    print("Statistical Analysis of Video Frames")
    print("1. Array-based approach (equivalent to stdDev.cpp)")
    print("2. OpenCV-based approach (equivalent to stdMan.cpp)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        results = calculate_pixel_statistics_array()
    elif choice == "2":
        results = calculate_pixel_statistics_opencv()
    else:
        print("Invalid choice. Using array-based approach...")
        results = calculate_pixel_statistics_array()
    
    if results:
        print(f"\nAnalysis completed successfully!")
        print(f"Processed {results['frames_processed']} frames")

if __name__ == "__main__":
    main()