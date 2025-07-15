import cv2
import numpy as np
import math

# Global variables
mouse_pos = (0, 0)
clicked = False
img_display = None
img_original = None

def rgb_to_hsv(r, g, b):
    """
    Manual RGB to HSV conversion (equivalent to original C++ implementation)
    """
    r, g, b = r/255.0, g/255.0, b/255.0
    
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    diff = max_val - min_val
    
    # Value
    v = max_val
    
    # Saturation
    if max_val == 0:
        s = 0
    else:
        s = diff / max_val
    
    # Hue
    if diff == 0:
        h = 0
    elif max_val == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif max_val == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    elif max_val == b:
        h = (60 * ((r - g) / diff) + 240) % 360
    
    return h, s * 255, v * 255

def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback function for interactive color selection
    """
    global mouse_pos, clicked, img_display, img_original
    
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pos = (x, y)
        clicked = True
        
        # Get BGR values at clicked position
        b, g, r = img_original[y, x]
        
        # Convert to HSV using manual conversion
        h, s, v = rgb_to_hsv(r, g, b)
        
        print(f"Clicked position: ({x}, {y})")
        print(f"BGR: ({b}, {g}, {r})")
        print(f"HSV: ({h:.1f}, {s:.1f}, {v:.1f})")
        
        # Convert image to HSV
        hsv_img = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)
        
        # Define HSV range for thresholding (with some tolerance)
        lower_bound = np.array([max(0, h-10), max(0, s-50), max(0, v-50)])
        upper_bound = np.array([min(179, h+10), min(255, s+50), min(255, v+50)])
        
        # Create mask for similar colors
        mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
        
        # Create colored output
        result = img_original.copy()
        result[mask == 255] = [0, 255, 255]  # Highlight similar colors in yellow
        
        img_display = result

def main():
    """
    Interactive color-based image segmentation using mouse clicks
    """
    global img_display, img_original
    
    # Try to load an image or use camera
    try:
        img_original = cv2.imread("test_image.jpg")
        if img_original is None:
            # If no image file, use camera
            cap = cv2.VideoCapture(0)
            ret, img_original = cap.read()
            cap.release()
            if not ret:
                print("Could not load image or capture from camera")
                return -1
    except:
        print("Using camera capture")
        cap = cv2.VideoCapture(0)
        ret, img_original = cap.read()
        cap.release()
        if not ret:
            print("Could not capture from camera")
            return -1
    
    img_display = img_original.copy()
    
    cv2.namedWindow("Click to segment colors")
    cv2.setMouseCallback("Click to segment colors", mouse_callback)
    
    print("Click on the image to segment similar colors. Press ESC to exit.")
    
    while True:
        cv2.imshow("Click to segment colors", img_display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == ord('r'):  # Reset image
            img_display = img_original.copy()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()