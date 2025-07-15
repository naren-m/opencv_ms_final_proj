import cv2
import numpy as np
import sys

def main():
    """
    Circle detection in images using Hough Transform
    """
    # Try to load image from command line argument or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Try some common image names or use camera
        image_path = "test_image.jpg"
    
    # Load image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Could not load image: {image_path}")
        print("Trying to capture from camera...")
        
        # Use camera if image loading fails
        cap = cv2.VideoCapture(0)
        ret, img = cap.read()
        cap.release()
        
        if not ret:
            print("Could not capture from camera")
            return -1
    
    # Create a copy for drawing
    output = img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Apply Hough Circle Transform
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1,              # Inverse ratio of accumulator resolution
        minDist=30,        # Minimum distance between circle centers
        param1=50,         # Upper threshold for edge detection
        param2=30,         # Accumulator threshold for center detection
        minRadius=5,       # Minimum circle radius
        maxRadius=100      # Maximum circle radius
    )
    
    # Draw detected circles
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        print(f"Detected {len(circles)} circles:")
        
        for i, (x, y, r) in enumerate(circles):
            # Draw circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            # Draw center
            cv2.circle(output, (x, y), 2, (0, 0, 255), 3)
            
            # Get pixel value at center (similar to cvGet2D in original)
            pixel_value = img[y, x]
            print(f"Circle {i+1}: Center=({x}, {y}), Radius={r}, Pixel BGR={pixel_value}")
    else:
        print("No circles detected")
    
    # Create windows and display results
    cv2.namedWindow("Original", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Grayscale", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Edges", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Detected Circles", cv2.WINDOW_AUTOSIZE)
    
    cv2.imshow("Original", img)
    cv2.imshow("Grayscale", gray)
    cv2.imshow("Edges", edges)
    cv2.imshow("Detected Circles", output)
    
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()