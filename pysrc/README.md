# OpenCV Computer Vision Project - Python Implementation

This repository contains Python implementations of various computer vision algorithms, converted from the original C++ OpenCV codebase. The project demonstrates fundamental computer vision techniques including object tracking, edge detection, shape recognition, face detection, and more.

## 📁 Project Structure

```
pysrc/
├── README.md
├── requirements.txt
├── object_tracking.py          # Main object tracking (converted from objectTracking.cpp)
├── task1/                      # Color analysis and interaction
│   ├── change_color.py         # HSV color space manipulation
│   ├── click_change.py         # Interactive color segmentation
│   ├── comp_hist.py           # Histogram comparison
│   ├── convert_hsv32.py       # High-precision HSV conversion
│   ├── circle_detection.py    # Hough circle detection
│   ├── histogram_utils.py     # Histogram visualization utilities
│   ├── template_matching.py   # Multi-method template matching
│   ├── mouse_rectangle.py     # Interactive rectangle drawing
│   └── motion_detection.py    # Frame differencing motion detection
├── task2/                      # Real-time histogram analysis
│   └── histogram_realtime.py  # Live color histogram visualization
├── task3/                      # Advanced image analysis
│   ├── face_detection.py      # Haar cascade face/eye detection
│   ├── background_subtraction_color.py    # Multi-channel background subtraction
│   ├── background_subtraction_simple.py   # Grayscale background subtraction
│   └── statistical_analysis.py            # Pixel variance analysis
├── task4/                      # Edge detection and shape analysis
│   ├── canny_edge_detection.py        # Interactive Canny edge detection
│   ├── square_detection.py            # Geometric square detection
│   ├── hough_line_detection.py        # Line detection algorithms
│   ├── sobel_edge_detection.py        # Sobel gradient analysis
│   ├── image_filtering.py             # Custom kernel filtering
│   └── pyramid_processing.py          # Image pyramid operations
├── expshapes/                  # Shape detection experiments
│   └── shape_detection.py     # Purple/blue rectangle detection
└── algorithms/                 # Core algorithms
    └── hsv_conversion.py       # Manual HSV conversion implementation
```

## 🚀 Quick Start

### Option 1: Docker (Recommended)

1. **Quick Docker setup**
   ```bash
   make quick-start     # Build Docker image and setup
   make docker-run      # Run with camera access
   ```

2. **Development with Docker**
   ```bash
   make docker-dev      # Interactive development container
   make jupyter         # Start Jupyter notebook server
   ```

### Option 2: Local Installation

1. **Setup local environment**
   ```bash
   make install         # Install dependencies and create directories
   ```

2. **Run applications**
   ```bash
   make run             # Main object tracking
   make run-face-detection
   make run-edge-detection
   ```

### Option 3: Manual Installation

1. **Clone the repository**
   ```bash
   cd opencv_ms_final_proj/pysrc
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python object_tracking.py
   ```

### Basic Usage

Most scripts support multiple modes:

```bash
# Real-time camera mode (default)
python script_name.py

# Static image mode
python script_name.py path/to/image.jpg

# Special modes (varies by script)
python script_name.py test
python script_name.py compare path/to/image.jpg
```

## 📋 Feature Overview

### 🎯 Object Tracking
- **File**: `object_tracking.py`
- **Features**: Real-time yellow object tracking with moment calculation
- **Usage**: `python object_tracking.py`

### 🎨 Color Analysis (task1/)

| Script | Description | Key Features |
|--------|-------------|--------------|
| `change_color.py` | HSV color space visualization | Channel separation, real-time conversion |
| `click_change.py` | Interactive color segmentation | Mouse-based color selection, HSV thresholding |
| `comp_hist.py` | Histogram comparison | Frame-to-frame correlation analysis |
| `circle_detection.py` | Hough circle detection | Multi-scale circle finding |
| `template_matching.py` | Template matching | 6 different correlation methods |

### 📊 Histogram Analysis (task2/)

| Script | Description | Usage |
|--------|-------------|-------|
| `histogram_realtime.py` | Live BGR histogram visualization | `python histogram_realtime.py` |
| | Static image analysis | `python histogram_realtime.py static [image_path]` |

### 🔍 Advanced Analysis (task3/)

| Script | Description | Key Algorithms |
|--------|-------------|----------------|
| `face_detection.py` | Face and eye detection | Haar cascades, multi-scale detection |
| `background_subtraction_color.py` | Multi-channel motion detection | RGB channel analysis, morphological ops |
| `statistical_analysis.py` | Pixel variance analysis | Frame differencing statistics |

### 📐 Edge Detection & Shapes (task4/)

| Script | Description | Techniques |
|--------|-------------|------------|
| `canny_edge_detection.py` | Interactive edge detection | Adjustable thresholds, real-time |
| `square_detection.py` | Geometric shape detection | Contour analysis, angle validation |
| `hough_line_detection.py` | Line detection | Standard & probabilistic Hough |
| `sobel_edge_detection.py` | Gradient-based edge detection | X/Y gradients, magnitude calculation |
| `image_filtering.py` | Custom kernel filtering | Multiple filter types, morphological ops |

## 🎮 Interactive Controls

Most applications support these common controls:

| Key | Action |
|-----|--------|
| `ESC` | Exit application |
| `s` | Save current frame/results |
| `h` | Show help information |
| `r` | Reset/restart |
| `t` | Toggle options or adjust thresholds |

## 📖 Detailed Usage Examples

### Object Tracking
```bash
# Basic object tracking (yellow objects)
python object_tracking.py

# The application will:
# - Open camera feed
# - Detect yellow objects using HSV thresholding
# - Track object centroids using image moments
# - Draw tracking trails
```

### Face Detection
```bash
# Real-time face detection
python task3/face_detection.py

# Static image face detection
python task3/face_detection.py path/to/photo.jpg

# Features:
# - Detects faces and eyes using Haar cascades
# - Draws ellipses around faces, circles around eyes
# - Shows detection statistics
```

### Interactive Edge Detection
```bash
# Canny edge detection with real-time parameter adjustment
python task4/canny_edge_detection.py

# Use trackbars to adjust:
# - Low threshold (0-255)
# - High threshold (0-255)
# - See results in real-time
```

### Shape Detection
```bash
# Square detection in real-time
python task4/square_detection.py

# Test with synthetic shapes
python task4/square_detection.py test

# Static image analysis
python task4/square_detection.py path/to/image.jpg
```

### HSV Color Analysis
```bash
# Compare HSV conversion methods
python algorithms/hsv_conversion.py

# Test conversion accuracy
python algorithms/hsv_conversion.py test

# Analyze image HSV properties
python algorithms/hsv_conversion.py path/to/image.jpg
```

## 🔧 Algorithm Details

### Color Tracking Algorithm
1. **HSV Conversion**: Convert BGR to HSV color space
2. **Color Thresholding**: Apply HSV range filtering
3. **Morphological Operations**: Clean up noise using erosion/dilation
4. **Moment Calculation**: Find object centroids using image moments
5. **Trail Drawing**: Connect previous and current positions

### Face Detection Pipeline
1. **Preprocessing**: Convert to grayscale, histogram equalization
2. **Multi-scale Detection**: Apply Haar cascades at different scales
3. **Non-maximum Suppression**: Remove overlapping detections
4. **Eye Detection**: Search for eyes within detected face regions
5. **Visualization**: Draw detection results with labels

### Edge Detection Methods
- **Canny**: Two-threshold edge detection with hysteresis
- **Sobel**: Gradient-based edge detection (X and Y directions)
- **Custom Kernels**: Various edge detection filters
- **Morphological**: Structure-based edge enhancement

## 🎯 Performance Tips

1. **Camera Resolution**: Reduce resolution for faster processing:
   ```python
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   ```

2. **Processing ROI**: Process only regions of interest:
   ```python
   roi = frame[y1:y2, x1:x2]
   ```

3. **Optimize Thresholds**: Adjust detection thresholds for your environment

4. **Use Appropriate Algorithms**: Choose faster algorithms for real-time applications

## 🐳 Docker Usage

### Docker Commands
```bash
# Build and run
make docker-build        # Build production image
make docker-build-dev    # Build development image
make docker-run          # Run with camera access
make docker-dev          # Development shell

# Docker Compose
make docker-compose-up   # Start all services
make docker-compose-dev  # Development environment
make jupyter             # Jupyter notebook server

# Utilities
make setup-x11           # Enable GUI on Linux
make test-camera         # Test camera access
```

### Docker Services
- **opencv-dev**: Development environment with full tools
- **opencv-prod**: Production container
- **jupyter**: Jupyter notebook server (port 8888)
- **opencv-web**: Web interface (port 8080)

## 🛠 Make Commands

```bash
# Quick Start
make help               # Show all available commands
make install           # Local development setup
make docker-install    # Docker development setup

# Development
make run               # Run main application
make run-face-detection # Face detection
make run-edge-detection # Edge detection  
make test              # Run tests
make lint              # Code linting
make format            # Code formatting

# Docker
make docker-build      # Build images
make docker-run        # Run container
make docker-dev        # Development container

# Utilities  
make clean             # Clean generated files
make info              # Project information
make list-apps         # List all applications
```

## 🐛 Troubleshooting

### Common Issues

1. **Camera not detected**
   ```bash
   make test-camera     # Test camera with Make
   # Or manually:
   python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
   ```

2. **Docker GUI issues (Linux)**
   ```bash
   make setup-x11       # Enable X11 forwarding
   ```

3. **Haar cascade files not found**
   ```python
   # Check OpenCV installation
   import cv2
   print(cv2.data.haarcascades)
   ```

4. **Performance issues**
   - Use Docker for consistent environment
   - Reduce image resolution
   - Close unnecessary windows
   - Use simpler algorithms for real-time processing

5. **Import errors**
   ```bash
   make clean           # Clean Python cache
   make setup           # Reinstall dependencies
   ```

### Docker-specific Issues

1. **Camera access denied**
   ```bash
   # Ensure camera device exists
   ls /dev/video*
   # Run with proper permissions
   make docker-run
   ```

2. **GUI not working**
   ```bash
   # On Linux
   make setup-x11
   echo $DISPLAY        # Should show display value
   ```

## 📚 Learning Resources

### Computer Vision Concepts
- **Color Spaces**: Understanding BGR, HSV, and LAB color representations
- **Image Moments**: Mathematical properties for object analysis
- **Morphological Operations**: Structure-based image processing
- **Feature Detection**: Corner, edge, and blob detection algorithms
- **Template Matching**: Pattern recognition techniques

### OpenCV Functions Used
- `cv2.cvtColor()`: Color space conversions
- `cv2.inRange()`: Color-based thresholding
- `cv2.moments()`: Image moment calculation
- `cv2.findContours()`: Contour detection
- `cv2.HoughCircles()`: Circle detection
- `cv2.Canny()`: Edge detection
- `cv2.matchTemplate()`: Template matching

## 🤝 Contributing

This project is educational in nature. Feel free to:
- Experiment with parameters
- Add new algorithms
- Improve existing implementations
- Create additional visualization tools

## 📄 License

This project is for educational purposes. Original C++ algorithms have been faithfully converted to Python while adding modern improvements and interactive features.

## 🔗 Related Projects

- Original C++ implementation (parent directory)
- OpenCV Documentation: https://docs.opencv.org/
- Computer Vision tutorials and examples

---

**Note**: This Python implementation maintains compatibility with the original C++ algorithms while leveraging Python's ease of use and modern libraries for enhanced functionality and visualization.