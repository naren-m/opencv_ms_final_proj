# OpenCV Project C++ Code Archive

## Archive Date
July 15, 2025

## Purpose
This archive contains all original C++ source code, headers, makefiles, and compiled objects from the OpenCV final project. The C++ code has been moved here to maintain a clean project structure while preserving the original implementation for reference.

## Archived Contents

### Root Level Files
- `DisplayImage.cpp` - Main display image functionality
- `DisplayImage.cpp~` - Backup of DisplayImage.cpp
- `coloridentification.cpp` - Color identification algorithms
- `objectTracking.cpp` - Object tracking implementation
- `objectTracking.cpp~` - Backup of objectTracking.cpp
- `objectTracking.o` - Compiled object file for object tracking
- `objectTracking2.cpp` - Second version of object tracking
- `objectTracking2.cpp~` - Backup of objectTracking2.cpp
- `makefile` - Root level makefile for compilation
- `obj` - Compiled object file
- `test` - Test executable

### Directory Structure Preserved

#### `cpp_original/algorithms/`
- `convertBgr2Hsv32.cpp` - BGR to HSV color space conversion
- `convertBgr2Hsv32.h` - Header for BGR to HSV conversion
- `convertBgr2Hsv32.h~` - Backup header file
- `#convertBgr2Hsv32.cpp#` - Emacs temporary file

#### `cpp_original/task1/`
Contains C++ implementations for Task 1 requirements:
- `changeColor.cpp` - Color changing functionality
- `clickChange.cpp` - Click-based color changes
- `compHist.cpp` - Histogram comparison
- `convertHsv32.cpp` - HSV conversion utilities
- `example.cpp` - Example implementations
- `functions.h` - Function declarations
- `hist.cpp` - Histogram processing
- `mouse.cpp` - Mouse interaction handling
- `seqFrames.cpp` - Sequential frame processing
- `seqFrames.cpp~` - Backup of seqFrames.cpp
- `makefile` - Task 1 specific makefile
- Various executables: `chg`, `cvt`, `seq`, `tempseq`, `test`
- `#tempseq#` - Emacs temporary file

#### `cpp_original/task2/`
Contains C++ implementations for Task 2 requirements:
- `hist.cpp` - Histogram processing for task 2
- `hist.cpp~` - Backup of hist.cpp
- `hist` - Compiled histogram executable

#### `cpp_original/task3/`
Contains C++ implementations for Task 3 requirements:
- `backChange.cpp` - Background change detection
- `backChange.cpp~` - Backup of backChange.cpp
- `change.cpp` - General change detection
- `faceDetect.cpp` - Face detection implementation
- `stdDev.cpp` - Standard deviation calculations
- `stdDev.cpp~` - Backup of stdDev.cpp
- `stdMan.cpp` - Statistical analysis
- `test` - Test executable
- `#faceDetect.cpp#` - Emacs temporary file

#### `cpp_original/task4/`
Contains C++ implementations for Task 4 requirements:
- `cannyTest.cpp` - Canny edge detection testing
- `copyworking.cpp` - Working copy of implementations
- `copyworking.cpp~` - Backup of copyworking.cpp
- `edgeDetection.cpp` - Edge detection algorithms
- `experimenting.cpp` - Experimental implementations
- `filter2d.cpp` - 2D filtering operations
- `filter2d.cpp~` - Backup of filter2d.cpp
- `houghLines.cpp` - Hough line detection
- `houghLines.cpp~` - Backup of houghLines.cpp
- `pyramid.cpp` - Pyramid processing
- `shapes.cpp` - Shape detection
- `sober.cpp` - Sobel edge detection
- `working.cpp` - Main working implementation
- Executables: `canny`, `test`
- Output files: `out.jpg`, `square.jpg`

#### `cpp_original/expshapes/`
Experimental shape detection code:
- `expshapes.cpp` - Main experimental shapes code
- `expshapes.cpp~` - Backup of expshapes.cpp
- `backupexpshapes.cpp` - Backup implementation
- `test.cpp` - Testing code
- `working.cpp` - Working implementation
- `working.cpp~` - Backup of working.cpp
- `test` - Test executable
- `1.png` - Test image

#### `cpp_original/opengl/`
OpenGL integration and visualization code:
- `opengltest.cpp` - OpenGL testing
- `opengltestforvideo.cpp` - OpenGL video processing
- `opengltestforvideo.cpp~` - Backup of video processing
- `camTest.cpp` - Camera testing
- `camTexture.cpp` - Camera texture handling
- `coloridentification.cpp` - Color ID for OpenGL
- `texture.cpp` - Texture processing
- `working.cpp` - Working OpenGL implementation
- `backupFinal.cpp` - Final backup implementation
- `test.cpp` - Testing code
- Executables: `camTexture`, `cube`, `opengl`, `openglTest`, `test`
- Media files: `1.png`, `lk_image.jpg`, `ram.jpg`, `test.avi`

## Python Implementation
The Python implementations remain in the `pysrc/` directory and were NOT moved to this archive. The Python code provides equivalent functionality and is the active codebase for this project.

## File Naming Conventions
- `.cpp~` files are backup files created by text editors
- `#filename#` files are temporary files created by Emacs
- Executables without extensions are compiled C++ programs
- `.o` files are compiled object files

## Compilation Notes
Each subdirectory contains its own makefile where applicable. The root makefile was used for general compilation tasks. All compiled executables and object files have been preserved in their original locations within the archive.

## Archive Structure
```
archive/
├── README.md (this file)
└── cpp_original/
    ├── [root level C++ files and executables]
    ├── algorithms/
    ├── task1/
    ├── task2/
    ├── task3/
    ├── task4/
    ├── expshapes/
    └── opengl/
```

## Notes
- All original directory structures have been preserved
- Backup files (.cpp~, .h~) have been maintained for reference
- Temporary editor files have been preserved as they may contain important development history
- All compiled executables and object files are included for completeness
- The Python codebase in `pysrc/` remains active and untouched