import cv2
import numpy as np
import os

def main():
    """
    Real-time face and eye detection from webcam feed using Haar cascade classifiers
    """
    # Load Haar cascade classifiers
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
    eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
    
    # Alternative paths if the above don't work
    if not os.path.exists(face_cascade_path):
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    if not os.path.exists(eye_cascade_path):
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
    
    # Load cascades
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    
    if face_cascade.empty():
        print("Error: Could not load face cascade classifier")
        return -1
    
    if eye_cascade.empty():
        print("Error: Could not load eye cascade classifier")
        print("Continuing with face detection only...")
        eye_detection_enabled = False
    else:
        eye_detection_enabled = True
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not initialize camera")
        return -1
    
    cv2.namedWindow("Face Detection")
    
    print("Face detection started. Press ESC to exit.")
    if eye_detection_enabled:
        print("Face and eye detection enabled.")
    else:
        print("Only face detection enabled.")
    
    frame_count = 0
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better contrast
        gray_equalized = cv2.equalizeHist(gray)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray_equalized,
            scaleFactor=1.1,      # How much the image size is reduced at each scale
            minNeighbors=5,       # How many neighbors each face should have
            minSize=(30, 30),     # Minimum face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        face_count = 0
        eye_count = 0
        
        # Draw faces and detect eyes
        for (x, y, w, h) in faces:
            face_count += 1
            
            # Draw ellipse around face (similar to original C++ code)
            center = (x + w//2, y + h//2)
            axes = (w//2, h//2)
            cv2.ellipse(frame, center, axes, 0, 0, 360, (255, 0, 255), 2)
            
            # Add face label
            cv2.putText(frame, f"Face {face_count}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            if eye_detection_enabled:
                # Region of interest for eye detection (within face)
                roi_gray = gray_equalized[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                
                # Detect eyes within face region
                eyes = eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(5, 5)
                )
                
                # Draw circles around eyes
                for (ex, ey, ew, eh) in eyes:
                    eye_count += 1
                    eye_center = (ex + ew//2, ey + eh//2)
                    radius = int(round((ew + eh) * 0.25))
                    cv2.circle(roi_color, eye_center, radius, (0, 255, 0), 2)
                    
                    # Add eye label
                    cv2.putText(roi_color, f"Eye", (ex, ey-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        # Add statistics overlay
        cv2.putText(frame, f"Frame: {frame_count}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Faces: {face_count}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if eye_detection_enabled:
            cv2.putText(frame, f"Eyes: {eye_count}", (10, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display result
        cv2.imshow("Face Detection", frame)
        
        # Print detection info
        if face_count > 0:
            if eye_detection_enabled:
                print(f"Frame {frame_count}: {face_count} face(s), {eye_count} eye(s) detected")
            else:
                print(f"Frame {frame_count}: {face_count} face(s) detected")
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == ord('s'):  # Save current frame
            cv2.imwrite(f"face_detection_frame_{frame_count}.jpg", frame)
            print(f"Frame {frame_count} saved")
        elif key == ord('g'):  # Show grayscale and equalized versions
            cv2.imshow("Grayscale", gray)
            cv2.imshow("Equalized", gray_equalized)
        elif key == ord('h'):  # Hide extra windows
            cv2.destroyWindow("Grayscale")
            cv2.destroyWindow("Equalized")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total frames processed: {frame_count}")

def detect_faces_in_image(image_path):
    """
    Detect faces in a static image file
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Load face cascade
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
    if not os.path.exists(face_cascade_path):
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    if face_cascade.empty():
        print("Error: Could not load face cascade classifier")
        return
    
    # Convert to grayscale and equalize
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_equalized = cv2.equalizeHist(gray)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray_equalized, 1.1, 5)
    
    # Draw detected faces
    for (x, y, w, h) in faces:
        center = (x + w//2, y + h//2)
        axes = (w//2, h//2)
        cv2.ellipse(img, center, axes, 0, 0, 360, (255, 0, 255), 2)
    
    print(f"Detected {len(faces)} face(s) in {image_path}")
    
    # Display result
    cv2.imshow("Face Detection - Static Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Static image mode
        detect_faces_in_image(sys.argv[1])
    else:
        # Real-time camera mode
        main()