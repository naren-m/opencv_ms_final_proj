import cv2
import numpy as np
import sys

def main():
    """
    Template matching using multiple correlation methods
    """
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: python template_matching.py <source_image> <template_image>")
        print("Using camera capture and creating template from center region...")
        
        # Use camera if no images provided
        cap = cv2.VideoCapture(0)
        ret, source = cap.read()
        cap.release()
        
        if not ret:
            print("Could not capture from camera")
            return -1
        
        # Create template from center region of captured image
        h, w = source.shape[:2]
        template = source[h//4:3*h//4, w//4:3*w//4]
    else:
        # Load images from command line arguments
        source = cv2.imread(sys.argv[1])
        template = cv2.imread(sys.argv[2])
        
        if source is None:
            print(f"Could not load source image: {sys.argv[1]}")
            return -1
        if template is None:
            print(f"Could not load template image: {sys.argv[2]}")
            return -1
    
    # Template matching methods (equivalent to original C++ implementation)
    methods = [
        cv2.TM_SQDIFF,
        cv2.TM_SQDIFF_NORMED,
        cv2.TM_CCORR,
        cv2.TM_CCORR_NORMED,
        cv2.TM_CCOEFF,
        cv2.TM_CCOEFF_NORMED
    ]
    
    method_names = [
        "TM_SQDIFF",
        "TM_SQDIFF_NORMED", 
        "TM_CCORR",
        "TM_CCORR_NORMED",
        "TM_CCOEFF",
        "TM_CCOEFF_NORMED"
    ]
    
    # Create windows
    cv2.namedWindow("Source Image", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Template", cv2.WINDOW_AUTOSIZE)
    
    cv2.imshow("Source Image", source)
    cv2.imshow("Template", template)
    
    # Perform template matching with each method
    results = []
    for i, method in enumerate(methods):
        # Perform template matching
        result = cv2.matchTemplate(source, template, method)
        
        # Normalize result to 0-255 range for visualization
        result_normalized = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Find best match location
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # For SQDIFF methods, minimum value is the best match
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            best_match = min_loc
            match_value = min_val
        else:
            best_match = max_loc
            match_value = max_val
        
        results.append({
            'method': method_names[i],
            'result': result_normalized,
            'best_match': best_match,
            'match_value': match_value,
            'min_val': min_val,
            'max_val': max_val
        })
        
        print(f"{method_names[i]}:")
        print(f"  Best match at: {best_match}")
        print(f"  Match value: {match_value:.4f}")
        print(f"  Min value: {min_val:.4f}, Max value: {max_val:.4f}")
        print()
    
    # Display results for each method
    current_method = 0
    
    def update_display():
        result_data = results[current_method]
        method_name = result_data['method']
        result_img = result_data['result']
        best_match = result_data['best_match']
        
        # Create a copy of source image and draw rectangle at best match
        source_with_rect = source.copy()
        template_h, template_w = template.shape[:2]
        top_left = best_match
        bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
        
        cv2.rectangle(source_with_rect, top_left, bottom_right, (0, 255, 0), 2)
        
        # Display images
        cv2.imshow("Source Image", source_with_rect)
        cv2.imshow(f"Match Result - {method_name}", result_img)
        
        print(f"Showing {method_name} (Method {current_method + 1}/{len(methods)})")
        print("Press 'n' for next method, 'p' for previous method, 'ESC' to exit")
    
    # Show initial result
    update_display()
    
    # Main loop for method switching
    while True:
        key = cv2.waitKey(0) & 0xFF
        
        if key == 27:  # ESC key
            break
        elif key == ord('n'):  # Next method
            current_method = (current_method + 1) % len(methods)
            update_display()
        elif key == ord('p'):  # Previous method
            current_method = (current_method - 1) % len(methods)
            update_display()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()