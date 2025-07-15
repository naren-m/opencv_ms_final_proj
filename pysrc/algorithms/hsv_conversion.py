import cv2
import numpy as np
import math

def convert_bgr_to_hsv_manual(image):
    """
    Manual BGR to HSV conversion implementation
    Equivalent to convertImageRGBtoHSV function in C++
    """
    # Get image dimensions
    height, width, channels = image.shape
    
    if channels != 3:
        raise ValueError("Input image must have 3 channels (BGR)")
    
    # Create output HSV image
    hsv_image = np.zeros_like(image, dtype=np.float32)
    
    # Constants
    BYTE_TO_FLOAT = 1.0 / 255.0
    
    for y in range(height):
        for x in range(width):
            # Get BGR pixel components (OpenCV stores as BGR)
            b, g, r = image[y, x]
            
            # Convert to float (0.0 to 1.0 range)
            fR = r * BYTE_TO_FLOAT
            fG = g * BYTE_TO_FLOAT
            fB = b * BYTE_TO_FLOAT
            
            # Find min and max values for HSV conversion
            # Use integer comparisons for slight speedup (as in original)
            if b < g:
                if b < r:
                    fMin = fB
                    if r > g:
                        iMax = r
                        fMax = fR
                    else:
                        iMax = g
                        fMax = fG
                else:
                    fMin = fR
                    fMax = fG
                    iMax = g
            else:
                if g < r:
                    fMin = fG
                    if b > r:
                        fMax = fB
                        iMax = b
                    else:
                        fMax = fR
                        iMax = r
                else:
                    fMin = fR
                    fMax = fB
                    iMax = b
            
            fDelta = fMax - fMin
            fV = fMax  # Value (Brightness)
            
            if iMax != 0:  # Not pure black
                fS = fDelta / fMax  # Saturation
                
                # Calculate Hue
                ANGLE_TO_UNIT = 1.0 / (6.0 * fDelta)  # Make Hues between 0.0 to 1.0
                
                if iMax == r:  # Between yellow and magenta
                    fH = (fG - fB) * ANGLE_TO_UNIT
                elif iMax == g:  # Between cyan and yellow
                    fH = (2.0/6.0) + (fB - fR) * ANGLE_TO_UNIT
                else:  # Between magenta and cyan
                    fH = (4.0/6.0) + (fR - fG) * ANGLE_TO_UNIT
                
                # Wrap outlier Hues around the circle
                if fH < 0.0:
                    fH += 1.0
                if fH >= 1.0:
                    fH -= 1.0
            else:
                # Pure black
                fS = 0.0
                fH = 0.0  # Undefined hue
            
            # Store HSV values (normalized 0-1 range)
            hsv_image[y, x] = [fH, fS, fV]
    
    return hsv_image

def convert_hsv_to_bgr_manual(hsv_image):
    """
    Manual HSV to BGR conversion implementation  
    Equivalent to convertImageHSVtoRGB function in C++
    """
    # Get image dimensions
    height, width, channels = hsv_image.shape
    
    if channels != 3:
        raise ValueError("Input image must have 3 channels (HSV)")
    
    # Create output BGR image
    bgr_image = np.zeros_like(hsv_image, dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            # Get HSV pixel components (assuming 0-1 range)
            fH, fS, fV = hsv_image[y, x]
            
            if fS == 0:  # Achromatic (grayscale)
                fR = fG = fB = fV
            else:
                # If Hue == 1.0, wrap it around to 0.0
                if fH >= 1.0:
                    fH = 0.0
                
                fH *= 6.0  # Scale to sector 0-5
                fI = math.floor(fH)  # Integer part
                iI = int(fH)
                fF = fH - fI  # Fractional part
                
                p = fV * (1.0 - fS)
                q = fV * (1.0 - fS * fF)
                t = fV * (1.0 - fS * (1.0 - fF))
                
                # Convert based on hue sector
                if iI == 0:
                    fR, fG, fB = fV, t, p
                elif iI == 1:
                    fR, fG, fB = q, fV, p
                elif iI == 2:
                    fR, fG, fB = p, fV, t
                elif iI == 3:
                    fR, fG, fB = p, q, fV
                elif iI == 4:
                    fR, fG, fB = t, p, fV
                else:  # iI == 5 or 6
                    fR, fG, fB = fV, p, q
            
            # Convert to 8-bit integers and clip
            bR = int(fR * 255.0)
            bG = int(fG * 255.0)
            bB = int(fB * 255.0)
            
            # Clip values to 0-255 range
            bR = max(0, min(255, bR))
            bG = max(0, min(255, bG))
            bB = max(0, min(255, bB))
            
            # Store BGR values (OpenCV format)
            bgr_image[y, x] = [bB, bG, bR]
    
    return bgr_image

def convert_bgr_to_hsv_numpy(image):
    """
    Vectorized NumPy implementation of BGR to HSV conversion
    Much faster than pixel-by-pixel conversion
    """
    # Normalize to 0-1 range
    image_float = image.astype(np.float32) / 255.0
    
    # Split BGR channels
    b, g, r = cv2.split(image_float)
    
    # Find min and max values
    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)
    delta = max_val - min_val
    
    # Initialize HSV channels
    h = np.zeros_like(max_val)
    s = np.zeros_like(max_val)
    v = max_val
    
    # Calculate saturation
    non_zero_mask = max_val != 0
    s[non_zero_mask] = delta[non_zero_mask] / max_val[non_zero_mask]
    
    # Calculate hue
    delta_non_zero = delta != 0
    
    # Red is max
    r_max = (max_val == r) & delta_non_zero
    h[r_max] = (g[r_max] - b[r_max]) / delta[r_max]
    
    # Green is max
    g_max = (max_val == g) & delta_non_zero
    h[g_max] = 2.0 + (b[g_max] - r[g_max]) / delta[g_max]
    
    # Blue is max
    b_max = (max_val == b) & delta_non_zero
    h[b_max] = 4.0 + (r[b_max] - g[b_max]) / delta[b_max]
    
    # Normalize hue to 0-1 range
    h = h / 6.0
    h[h < 0] += 1.0
    h[h >= 1.0] -= 1.0
    
    # Merge channels
    hsv_image = cv2.merge([h, s, v])
    
    return hsv_image

def compare_hsv_methods():
    """
    Compare different HSV conversion methods: manual, NumPy, and OpenCV
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not initialize camera")
        return -1
    
    cv2.namedWindow("Original")
    cv2.namedWindow("Manual HSV")
    cv2.namedWindow("NumPy HSV")
    cv2.namedWindow("OpenCV HSV")
    cv2.namedWindow("Conversion Comparison")
    
    print("HSV conversion methods comparison started.")
    print("Press ESC to exit, 's' to save, 't' to toggle timing")
    
    show_timing = False
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Resize frame for faster processing (optional)
        frame_small = cv2.resize(frame, (320, 240))
        
        # Time the conversions if requested
        if show_timing:
            import time
            
            # Manual conversion timing
            start_time = time.time()
            hsv_manual = convert_bgr_to_hsv_manual(frame_small)
            manual_time = (time.time() - start_time) * 1000
            
            # NumPy conversion timing
            start_time = time.time()
            hsv_numpy = convert_bgr_to_hsv_numpy(frame_small)
            numpy_time = (time.time() - start_time) * 1000
            
            # OpenCV conversion timing
            start_time = time.time()
            hsv_opencv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
            opencv_time = (time.time() - start_time) * 1000
            
        else:
            # Regular conversions without timing
            hsv_manual = convert_bgr_to_hsv_manual(frame_small)
            hsv_numpy = convert_bgr_to_hsv_numpy(frame_small)
            hsv_opencv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
        
        # Convert HSV images to displayable format (0-255 range)
        hsv_manual_display = (hsv_manual * 255).astype(np.uint8)
        hsv_numpy_display = (hsv_numpy * 255).astype(np.uint8)
        hsv_opencv_display = (hsv_opencv * 255).astype(np.uint8)
        
        # Create comparison visualization
        # Convert HSV to BGR for better visualization
        hsv_manual_bgr = cv2.cvtColor(hsv_manual_display, cv2.COLOR_HSV2BGR)
        hsv_numpy_bgr = cv2.cvtColor(hsv_numpy_display, cv2.COLOR_HSV2BGR)
        hsv_opencv_bgr = cv2.cvtColor(hsv_opencv_display, cv2.COLOR_HSV2BGR)
        
        # Create side-by-side comparison
        top_row = np.hstack([frame_small, hsv_manual_bgr])
        bottom_row = np.hstack([hsv_numpy_bgr, hsv_opencv_bgr])
        comparison = np.vstack([top_row, bottom_row])
        
        # Add labels
        cv2.putText(comparison, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(comparison, "Manual HSV", (330, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(comparison, "NumPy HSV", (10, 270), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(comparison, "OpenCV HSV", (330, 270), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add timing information if enabled
        if show_timing:
            cv2.putText(comparison, f"Manual: {manual_time:.1f}ms", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(comparison, f"NumPy: {numpy_time:.1f}ms", (10, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(comparison, f"OpenCV: {opencv_time:.1f}ms", (330, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Display results
        cv2.imshow("Original", frame)
        cv2.imshow("Manual HSV", hsv_manual_display)
        cv2.imshow("NumPy HSV", hsv_numpy_display)
        cv2.imshow("OpenCV HSV", hsv_opencv_display)
        cv2.imshow("Conversion Comparison", comparison)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s'):  # Save current results
            cv2.imwrite(f"hsv_original_{frame_count}.jpg", frame)
            cv2.imwrite(f"hsv_manual_{frame_count}.jpg", hsv_manual_display)
            cv2.imwrite(f"hsv_numpy_{frame_count}.jpg", hsv_numpy_display)
            cv2.imwrite(f"hsv_opencv_{frame_count}.jpg", hsv_opencv_display)
            cv2.imwrite(f"hsv_comparison_{frame_count}.jpg", comparison)
            print(f"Frame {frame_count} saved")
        elif key == ord('t'):  # Toggle timing display
            show_timing = not show_timing
            print(f"Timing display: {'ON' if show_timing else 'OFF'}")
        elif key == ord('h'):  # Show help
            print("\nControls:")
            print("ESC - Exit")
            print("s - Save current frame and HSV conversions")
            print("t - Toggle timing display")
            print("h - Show this help")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total frames processed: {frame_count}")

def test_hsv_conversion_accuracy():
    """
    Test the accuracy of manual HSV conversion against OpenCV
    """
    # Create test image with known colors
    test_colors = np.array([
        [[255, 0, 0]],      # Pure red
        [[0, 255, 0]],      # Pure green
        [[0, 0, 255]],      # Pure blue
        [[255, 255, 0]],    # Yellow
        [[255, 0, 255]],    # Magenta
        [[0, 255, 255]],    # Cyan
        [[255, 255, 255]],  # White
        [[0, 0, 0]],        # Black
        [[128, 128, 128]]   # Gray
    ], dtype=np.uint8)
    
    print("Testing HSV conversion accuracy...")
    print("Color\t\tManual HSV\t\tOpenCV HSV\t\tDifference")
    print("-" * 80)
    
    color_names = ["Red", "Green", "Blue", "Yellow", "Magenta", "Cyan", "White", "Black", "Gray"]
    
    for i, color in enumerate(test_colors):
        # Manual conversion
        hsv_manual = convert_bgr_to_hsv_manual(color)
        hsv_manual_scaled = hsv_manual[0, 0] * np.array([179, 255, 255])  # Scale to OpenCV range
        
        # OpenCV conversion
        hsv_opencv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[0, 0]
        
        # Calculate difference
        diff = np.abs(hsv_manual_scaled - hsv_opencv)
        
        print(f"{color_names[i]:<10}\t[{hsv_manual_scaled[0]:6.1f},{hsv_manual_scaled[1]:6.1f},{hsv_manual_scaled[2]:6.1f}]\t"
              f"[{hsv_opencv[0]:6.1f},{hsv_opencv[1]:6.1f},{hsv_opencv[2]:6.1f}]\t"
              f"[{diff[0]:6.1f},{diff[1]:6.1f},{diff[2]:6.1f}]")

def analyze_hsv_image(image_path):
    """
    Analyze HSV properties of a static image
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"Analyzing HSV properties of: {image_path}")
    
    # Convert using different methods
    hsv_manual = convert_bgr_to_hsv_manual(img)
    hsv_opencv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    
    # Split HSV channels
    h_manual, s_manual, v_manual = cv2.split(hsv_manual)
    h_opencv, s_opencv, v_opencv = cv2.split(hsv_opencv)
    
    # Calculate statistics
    print("\nHSV Channel Statistics (Manual vs OpenCV):")
    print("Channel\t\tManual\t\t\tOpenCV")
    print("-------\t\t------\t\t\t------")
    
    channels = [("Hue", h_manual, h_opencv), 
                ("Saturation", s_manual, s_opencv), 
                ("Value", v_manual, v_opencv)]
    
    for name, manual, opencv in channels:
        print(f"{name}\t\tMean: {np.mean(manual):.3f}\t\tMean: {np.mean(opencv):.3f}")
        print(f"\t\tStd:  {np.std(manual):.3f}\t\tStd:  {np.std(opencv):.3f}")
        print(f"\t\tMin:  {np.min(manual):.3f}\t\tMin:  {np.min(opencv):.3f}")
        print(f"\t\tMax:  {np.max(manual):.3f}\t\tMax:  {np.max(opencv):.3f}")
        print()
    
    # Display results
    hsv_manual_display = (hsv_manual * 255).astype(np.uint8)
    hsv_opencv_display = (hsv_opencv * 255).astype(np.uint8)
    
    cv2.namedWindow("Original", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Manual HSV", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("OpenCV HSV", cv2.WINDOW_AUTOSIZE)
    
    cv2.imshow("Original", img)
    cv2.imshow("Manual HSV", hsv_manual_display)
    cv2.imshow("OpenCV HSV", hsv_opencv_display)
    
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # Real-time comparison mode
        compare_hsv_methods()
    elif sys.argv[1] == "test":
        # Accuracy test mode
        test_hsv_conversion_accuracy()
    elif len(sys.argv) == 2:
        # Static image analysis
        analyze_hsv_image(sys.argv[1])
    else:
        print("Usage:")
        print("  python hsv_conversion.py                     # Real-time comparison")
        print("  python hsv_conversion.py test                # Accuracy test")
        print("  python hsv_conversion.py <image_path>        # Static image analysis")