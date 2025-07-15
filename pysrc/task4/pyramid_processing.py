import cv2
import numpy as np

def main():
    """
    Image pyramid processing demonstration
    Equivalent to pyramid.cpp
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not initialize camera")
        return -1
    
    cv2.namedWindow("Original")
    cv2.namedWindow("Pyramid Down")
    cv2.namedWindow("Pyramid Up")
    cv2.namedWindow("Pyramid Down x2")
    cv2.namedWindow("Laplacian Pyramid")
    
    print("Image pyramid processing started.")
    print("Press ESC to exit, 's' to save current frame, 'h' for help")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Pyramid down - reduce image size by half
        pyr_down = cv2.pyrDown(frame)
        
        # Pyramid up - increase image size (from reduced image)
        pyr_up = cv2.pyrUp(pyr_down)
        
        # Resize pyrUp to match original size for display
        if pyr_up.shape[:2] != frame.shape[:2]:
            pyr_up = cv2.resize(pyr_up, (frame.shape[1], frame.shape[0]))
        
        # Pyramid down twice - reduce by factor of 4
        pyr_down_2x = cv2.pyrDown(pyr_down)
        
        # Create Laplacian pyramid level
        # Laplacian = Original - PyrUp(PyrDown(Original))
        laplacian = cv2.subtract(frame, pyr_up)
        
        # Add text overlays
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Size: {frame.shape[1]}x{frame.shape[0]}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(pyr_down, f"Size: {pyr_down.shape[1]}x{pyr_down.shape[0]}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(pyr_down, "Pyramid Down", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(pyr_up, "Pyramid Up", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(pyr_up, "(Reconstructed)", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(pyr_down_2x, f"Size: {pyr_down_2x.shape[1]}x{pyr_down_2x.shape[0]}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(pyr_down_2x, "Pyramid Down x2", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(laplacian, "Laplacian Pyramid", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display all pyramid levels
        cv2.imshow("Original", frame)
        cv2.imshow("Pyramid Down", pyr_down)
        cv2.imshow("Pyramid Up", pyr_up)
        cv2.imshow("Pyramid Down x2", pyr_down_2x)
        cv2.imshow("Laplacian Pyramid", laplacian)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s'):  # Save current pyramid levels
            cv2.imwrite(f"pyramid_original_{frame_count}.jpg", frame)
            cv2.imwrite(f"pyramid_down_{frame_count}.jpg", pyr_down)
            cv2.imwrite(f"pyramid_up_{frame_count}.jpg", pyr_up)
            cv2.imwrite(f"pyramid_down_2x_{frame_count}.jpg", pyr_down_2x)
            cv2.imwrite(f"pyramid_laplacian_{frame_count}.jpg", laplacian)
            print(f"Frame {frame_count} pyramid levels saved")
        elif key == ord('h'):  # Show help
            print("\nControls:")
            print("ESC - Exit")
            print("s - Save current pyramid levels")
            print("h - Show this help")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total frames processed: {frame_count}")

def create_gaussian_pyramid(image, levels=4):
    """
    Create a complete Gaussian pyramid
    """
    pyramid = [image]
    current = image.copy()
    
    for i in range(levels):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    
    return pyramid

def create_laplacian_pyramid(image, levels=4):
    """
    Create a Laplacian pyramid
    """
    # First create Gaussian pyramid
    gaussian_pyramid = create_gaussian_pyramid(image, levels)
    
    laplacian_pyramid = []
    
    for i in range(levels):
        # Get current and next level from Gaussian pyramid
        current_level = gaussian_pyramid[i]
        next_level = gaussian_pyramid[i + 1]
        
        # Expand next level to current level size
        expanded = cv2.pyrUp(next_level)
        
        # Resize to match exact dimensions if needed
        if expanded.shape[:2] != current_level.shape[:2]:
            expanded = cv2.resize(expanded, (current_level.shape[1], current_level.shape[0]))
        
        # Calculate Laplacian level
        laplacian_level = cv2.subtract(current_level, expanded)
        laplacian_pyramid.append(laplacian_level)
    
    # Add the smallest Gaussian level as the last Laplacian level
    laplacian_pyramid.append(gaussian_pyramid[levels])
    
    return laplacian_pyramid

def reconstruct_from_laplacian(laplacian_pyramid):
    """
    Reconstruct image from Laplacian pyramid
    """
    # Start with the smallest level (last element)
    current = laplacian_pyramid[-1].copy()
    
    # Reconstruct from smallest to largest
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        # Expand current level
        current = cv2.pyrUp(current)
        
        # Resize to match the target level if needed
        target_level = laplacian_pyramid[i]
        if current.shape[:2] != target_level.shape[:2]:
            current = cv2.resize(current, (target_level.shape[1], target_level.shape[0]))
        
        # Add the Laplacian level
        current = cv2.add(current, target_level)
    
    return current

def pyramid_blending(img1, img2, mask, levels=4):
    """
    Blend two images using pyramid blending technique
    """
    # Create Laplacian pyramids for both images
    lap_pyramid1 = create_laplacian_pyramid(img1, levels)
    lap_pyramid2 = create_laplacian_pyramid(img2, levels)
    
    # Create Gaussian pyramid for mask
    mask_pyramid = create_gaussian_pyramid(mask, levels)
    
    # Blend each level of the pyramids
    blended_pyramid = []
    for i in range(len(lap_pyramid1)):
        # Normalize mask to 0-1 range
        mask_level = mask_pyramid[i].astype(np.float32) / 255.0
        
        # Resize mask if it has fewer channels than images
        if len(mask_level.shape) == 2 and len(lap_pyramid1[i].shape) == 3:
            mask_level = cv2.merge([mask_level, mask_level, mask_level])
        
        # Resize mask to match pyramid level size
        if mask_level.shape[:2] != lap_pyramid1[i].shape[:2]:
            mask_level = cv2.resize(mask_level, (lap_pyramid1[i].shape[1], lap_pyramid1[i].shape[0]))
        
        # Blend using mask
        blended_level = (lap_pyramid1[i].astype(np.float32) * mask_level + 
                        lap_pyramid2[i].astype(np.float32) * (1.0 - mask_level))
        blended_pyramid.append(blended_level.astype(np.uint8))
    
    # Reconstruct the blended image
    blended_image = reconstruct_from_laplacian(blended_pyramid)
    
    return blended_image

def demonstrate_pyramid_operations(image_path):
    """
    Demonstrate various pyramid operations on a static image
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    
    # Create pyramids
    gaussian_pyramid = create_gaussian_pyramid(img, 4)
    laplacian_pyramid = create_laplacian_pyramid(img, 4)
    
    # Reconstruct image from Laplacian pyramid
    reconstructed = reconstruct_from_laplacian(laplacian_pyramid)
    
    # Display Gaussian pyramid
    print("Showing Gaussian pyramid levels...")
    for i, level in enumerate(gaussian_pyramid):
        cv2.namedWindow(f"Gaussian Level {i}", cv2.WINDOW_AUTOSIZE)
        cv2.imshow(f"Gaussian Level {i}", level)
    
    # Display Laplacian pyramid
    print("Showing Laplacian pyramid levels...")
    for i, level in enumerate(laplacian_pyramid):
        # Normalize Laplacian levels for display (they can have negative values)
        level_display = cv2.normalize(level, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.namedWindow(f"Laplacian Level {i}", cv2.WINDOW_AUTOSIZE)
        cv2.imshow(f"Laplacian Level {i}", level_display)
    
    # Show original vs reconstructed
    cv2.namedWindow("Original", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Reconstructed", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Original", img)
    cv2.imshow("Reconstructed", reconstructed)
    
    # Calculate reconstruction error
    diff = cv2.absdiff(img, reconstructed)
    cv2.namedWindow("Reconstruction Error", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Reconstruction Error", diff)
    
    error_mean = np.mean(diff)
    print(f"Reconstruction error (mean): {error_mean:.2f}")
    
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def pyramid_noise_reduction():
    """
    Demonstrate noise reduction using pyramid processing
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not initialize camera")
        return -1
    
    cv2.namedWindow("Original")
    cv2.namedWindow("Noise Reduced")
    cv2.namedWindow("Difference")
    
    print("Pyramid-based noise reduction started.")
    print("Press ESC to exit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply pyramid-based noise reduction
        # Method: pyrDown followed by pyrUp acts as a low-pass filter
        reduced = cv2.pyrDown(frame)
        noise_reduced = cv2.pyrUp(reduced)
        
        # Resize to match original if needed
        if noise_reduced.shape[:2] != frame.shape[:2]:
            noise_reduced = cv2.resize(noise_reduced, (frame.shape[1], frame.shape[0]))
        
        # Calculate difference (removed noise)
        diff = cv2.absdiff(frame, noise_reduced)
        
        # Display results
        cv2.imshow("Original", frame)
        cv2.imshow("Noise Reduced", noise_reduced)
        cv2.imshow("Difference", diff)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # Real-time camera mode
        main()
    elif len(sys.argv) == 2:
        # Static image pyramid demonstration
        demonstrate_pyramid_operations(sys.argv[1])
    elif len(sys.argv) == 2 and sys.argv[1] == "noise":
        # Noise reduction mode
        pyramid_noise_reduction()
    else:
        print("Usage:")
        print("  python pyramid_processing.py                     # Camera mode")
        print("  python pyramid_processing.py <image_path>        # Single image mode")
        print("  python pyramid_processing.py noise               # Noise reduction mode")