import cv2
import numpy as np

def create_custom_kernels():
    """
    Create various custom kernels for 2D filtering
    Based on the kernels from filter2d.cpp and experimenting.cpp
    """
    kernels = {}
    
    # Edge detection kernels from original C++ code
    kernels['edge_detection_1'] = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    
    kernels['edge_detection_2'] = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    
    # 5x5 edge detection kernel from edgeDetection.cpp
    kernels['edge_5x5'] = np.array([
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1],
        [-1, -1, 24, -1, -1],
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1]
    ], dtype=np.float32)
    
    # Sobel-like kernels
    kernels['sobel_x'] = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)
    
    kernels['sobel_y'] = np.array([
        [-1, -2, -1],
        [0,   0,  0],
        [1,   2,  1]
    ], dtype=np.float32)
    
    # Blur/smoothing kernels
    kernels['box_blur'] = np.ones((3, 3), dtype=np.float32) / 9
    
    kernels['gaussian_3x3'] = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float32) / 16
    
    kernels['gaussian_5x5'] = np.array([
        [1,  4,  6,  4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1,  4,  6,  4, 1]
    ], dtype=np.float32) / 256
    
    # Sharpening kernel
    kernels['sharpen'] = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    
    # Emboss kernel
    kernels['emboss'] = np.array([
        [-2, -1, 0],
        [-1,  1, 1],
        [0,   1, 2]
    ], dtype=np.float32)
    
    # High-pass filter
    kernels['high_pass'] = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    
    return kernels

def main():
    """
    Comprehensive demonstration of 2D filtering with various kernels
    Equivalent to filter2d.cpp and experimenting.cpp
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not initialize camera")
        return -1
    
    # Get available kernels
    kernels = create_custom_kernels()
    kernel_names = list(kernels.keys())
    current_kernel_idx = 0
    
    cv2.namedWindow("Original")
    cv2.namedWindow("Filtered")
    cv2.namedWindow("Kernel Info")
    
    print("2D Filtering with custom kernels started.")
    print("Press 'n' for next kernel, 'p' for previous kernel, ESC to exit")
    print(f"Available kernels: {', '.join(kernel_names)}")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Get current kernel
        current_kernel_name = kernel_names[current_kernel_idx]
        current_kernel = kernels[current_kernel_name]
        
        # Convert to grayscale for most filters
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply 2D filter
        filtered = cv2.filter2D(gray, -1, current_kernel)
        
        # Handle different data types and ranges
        if filtered.dtype != np.uint8:
            # Normalize to 0-255 range for display
            filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Convert back to BGR for display
        filtered_bgr = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
        
        # Add text overlays
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Kernel: {current_kernel_name}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(filtered_bgr, current_kernel_name, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Create kernel visualization
        kernel_info = create_kernel_visualization(current_kernel, current_kernel_name)
        
        # Display images
        cv2.imshow("Original", frame)
        cv2.imshow("Filtered", filtered_bgr)
        cv2.imshow("Kernel Info", kernel_info)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('n'):  # Next kernel
            current_kernel_idx = (current_kernel_idx + 1) % len(kernel_names)
            print(f"Switched to kernel: {kernel_names[current_kernel_idx]}")
        elif key == ord('p'):  # Previous kernel
            current_kernel_idx = (current_kernel_idx - 1) % len(kernel_names)
            print(f"Switched to kernel: {kernel_names[current_kernel_idx]}")
        elif key == ord('s'):  # Save current results
            cv2.imwrite(f"filter_original_{frame_count}.jpg", frame)
            cv2.imwrite(f"filter_{current_kernel_name}_{frame_count}.jpg", filtered)
            print(f"Frame {frame_count} with {current_kernel_name} filter saved")
        elif key == ord('i'):  # Print kernel info
            print(f"\nKernel: {current_kernel_name}")
            print(f"Size: {current_kernel.shape}")
            print(f"Values:\n{current_kernel}")
        elif key == ord('h'):  # Show help
            print("\nControls:")
            print("ESC - Exit")
            print("n - Next kernel")
            print("p - Previous kernel")
            print("s - Save current frame and filtered result")
            print("i - Print kernel information")
            print("h - Show this help")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total frames processed: {frame_count}")

def create_kernel_visualization(kernel, name):
    """
    Create a visual representation of the kernel
    """
    # Create a larger image for kernel display
    kernel_img = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Add kernel name
    cv2.putText(kernel_img, name, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add kernel size
    cv2.putText(kernel_img, f"Size: {kernel.shape[0]}x{kernel.shape[1]}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display kernel values as text
    start_y = 90
    for i, row in enumerate(kernel):
        row_text = " ".join([f"{val:6.2f}" for val in row])
        cv2.putText(kernel_img, row_text, (10, start_y + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Add kernel sum (useful for understanding the filter effect)
    kernel_sum = np.sum(kernel)
    cv2.putText(kernel_img, f"Sum: {kernel_sum:.2f}", (10, start_y + len(kernel) * 25 + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return kernel_img

def multi_stage_filtering(image):
    """
    Apply multiple filtering stages as in experimenting.cpp
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Stage 1: Median blur
    stage1 = cv2.medianBlur(gray, 5)
    
    # Stage 2: Custom edge detection
    edge_kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    
    stage2 = cv2.filter2D(stage1, -1, edge_kernel)
    
    # Stage 3: Another median blur
    stage3 = cv2.medianBlur(stage2, 3)
    
    # Stage 4: Cross-multiplication with another filter
    blur_kernel = np.ones((3, 3), dtype=np.float32) / 9
    stage4 = cv2.filter2D(stage1, -1, blur_kernel)
    
    # Combine results (experimental as in original)
    # Convert to float for calculations
    stage3_f = stage3.astype(np.float32)
    stage4_f = stage4.astype(np.float32)
    
    # Element-wise multiplication
    combined = stage3_f * stage4_f / 255.0  # Normalize
    combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    return stage1, stage2, stage3, stage4, combined

def filter_static_image(image_path):
    """
    Apply various filters to a static image
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    
    # Get kernels
    kernels = create_custom_kernels()
    
    # Apply multi-stage filtering
    stage1, stage2, stage3, stage4, combined = multi_stage_filtering(img)
    
    # Display multi-stage results
    cv2.namedWindow("Original", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Stage 1: Median Blur", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Stage 2: Edge Detection", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Stage 3: Median Blur 2", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Stage 4: Box Blur", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Combined Result", cv2.WINDOW_AUTOSIZE)
    
    cv2.imshow("Original", img)
    cv2.imshow("Stage 1: Median Blur", stage1)
    cv2.imshow("Stage 2: Edge Detection", stage2)
    cv2.imshow("Stage 3: Median Blur 2", stage3)
    cv2.imshow("Stage 4: Box Blur", stage4)
    cv2.imshow("Combined Result", combined)
    
    print("Multi-stage filtering complete.")
    print("Press any key to continue to individual kernel demonstration...")
    cv2.waitKey(0)
    
    # Demonstrate individual kernels
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    for kernel_name, kernel in kernels.items():
        filtered = cv2.filter2D(gray, -1, kernel)
        
        # Normalize for display
        if filtered.dtype != np.uint8:
            filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        cv2.imshow(f"Filter: {kernel_name}", filtered)
        print(f"Showing {kernel_name} filter. Press any key for next...")
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()

def compare_blur_methods(image_path):
    """
    Compare different blur/smoothing methods
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Different blur methods
    blur_box = cv2.blur(gray, (9, 9))
    blur_gaussian = cv2.GaussianBlur(gray, (9, 9), 0)
    blur_median = cv2.medianBlur(gray, 9)
    blur_bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Custom kernel blur
    kernels = create_custom_kernels()
    blur_custom = cv2.filter2D(gray, -1, kernels['gaussian_5x5'])
    
    # Create comparison
    results = [
        ("Original", gray),
        ("Box Blur", blur_box),
        ("Gaussian Blur", blur_gaussian),
        ("Median Blur", blur_median),
        ("Bilateral Filter", blur_bilateral),
        ("Custom Gaussian", blur_custom)
    ]
    
    # Display in grid
    for i, (name, result) in enumerate(results):
        cv2.putText(result, name, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(name, result)
    
    print("Blur methods comparison. Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # Real-time camera mode
        main()
    elif len(sys.argv) == 2:
        # Static image mode
        filter_static_image(sys.argv[1])
    elif len(sys.argv) == 3 and sys.argv[1] == "blur":
        # Blur comparison mode
        compare_blur_methods(sys.argv[2])
    else:
        print("Usage:")
        print("  python image_filtering.py                        # Camera mode")
        print("  python image_filtering.py <image_path>           # Single image mode")
        print("  python image_filtering.py blur <image_path>      # Blur comparison")