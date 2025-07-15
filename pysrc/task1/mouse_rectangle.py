import cv2
import numpy as np

# Global variables
drawing = False
start_point = (0, 0)
end_point = (0, 0)
img_display = None
img_original = None

def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback function for interactive rectangle drawing
    """
    global drawing, start_point, end_point, img_display, img_original
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        
        # Get and print pixel value at click position
        b, g, r = img_original[y, x]
        print(f"Clicked at ({x}, {y}) - BGR: ({b}, {g}, {r})")
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)
            # Create temporary image for preview
            img_display = img_original.copy()
            cv2.rectangle(img_display, start_point, end_point, (0, 255, 0), 2)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        
        # Draw final rectangle
        cv2.rectangle(img_display, start_point, end_point, (0, 255, 0), 2)
        
        # Calculate rectangle properties
        width = abs(end_point[0] - start_point[0])
        height = abs(end_point[1] - start_point[1])
        area = width * height
        
        print(f"Rectangle drawn from {start_point} to {end_point}")
        print(f"Width: {width}, Height: {height}, Area: {area}")
        
        # Get pixel values at rectangle corners
        corners = [start_point, end_point, 
                  (start_point[0], end_point[1]), 
                  (end_point[0], start_point[1])]
        
        for i, (cx, cy) in enumerate(corners):
            if 0 <= cy < img_original.shape[0] and 0 <= cx < img_original.shape[1]:
                b, g, r = img_original[cy, cx]
                print(f"Corner {i+1} ({cx}, {cy}) - BGR: ({b}, {g}, {r})")

def main():
    """
    Interactive rectangle drawing with mouse input
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
    
    cv2.namedWindow("Draw Rectangle")
    cv2.setMouseCallback("Draw Rectangle", mouse_callback)
    
    print("Draw rectangles by clicking and dragging with the mouse.")
    print("Press 'r' to reset, 'c' to clear all rectangles, 'ESC' to exit.")
    
    while True:
        cv2.imshow("Draw Rectangle", img_display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == ord('r'):  # Reset to original image
            img_display = img_original.copy()
            print("Image reset")
        elif key == ord('c'):  # Clear (same as reset in this simple version)
            img_display = img_original.copy()
            print("Rectangles cleared")
        elif key == ord('s'):  # Save current image
            cv2.imwrite("rectangle_output.jpg", img_display)
            print("Image saved as rectangle_output.jpg")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()