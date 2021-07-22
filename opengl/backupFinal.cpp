
/************************************************************************************************ 
REFERNECE STARTING WEBCAM: 

1) on console 
   if you have to reset you cam for any reason, do: 
   1. unplug your cam 
   2. modprobe -r uvcvideo 
   3. modprobe -r snd_usb_audio 
   4. replug your cam  
**************************************************************************************************/ 
#include<iostream> 
// openCV 
#include<cv.h> 
#include<cvaux.h> 
#include<highgui.h> 
// openGL & GLUI 
#include <GL/glut.h>   // into usr/include/GL/ 
#include <GL/gl.h> 
//#include "glui.h" // <GL/glui.h>   // into usr/include/GL/ 
// c 
#include <math.h> 

// C-Stuff for RS232 
#include <stdio.h> 
#include <stdlib.h> 

#include <fcntl.h> 
#include <string.h> 
#include <ctype.h> 

#include <setjmp.h> 
#include <termios.h> 
#include <unistd.h> 
#include <sys/signal.h> 
#include <sys/types.h> 
#include <errno.h> 

#include <ctype.h> 
#include <sys/io.h> 
// glui 
#include <sys/stat.h> 
#include <stdarg.h> 
#include <assert.h> 
#include <float.h> 
#include <limits.h> 
#include <time.h> 

#include <sstream> 
  
using namespace std; 


bool bool_makePic; 
bool quit; 
bool done; 

// stuff from lk_glDemo.cpp 
int height, width, counts, init; 
float fps, rot; 

CvCapture *capture = 0; 
//IplImage *image; 
GLenum format; 
GLuint imageID; 

#define IsRGB(s) ((s[0] == 'R') && (s[1] == 'G') && (s[2] == 'B')) 
#define IsBGR(s) ((s[0] == 'B') && (s[1] == 'G') && (s[2] == 'R')) 

#ifndef GL_CLAMP_TO_BORDER 
#define GL_CLAMP_TO_BORDER 0x812D 
#endif 
#define GL_MIRROR_CLAMP_EXT 0x8742 

IplImage *image = 0, *grey = 0, *prev_grey = 0, *pyramid = 0, *prev_pyramid = 0, *swap_temp; 

/// 
IplImage* trace; 
/// 

int win_size = 10; 
const int MAX_COUNT = 500; 
CvPoint2D32f* points[2] = {0,0}, *swap_points; 
char* status = 0; 
int countLK = 0; 
int need_to_init = 0; 
int night_mode = 0; 
int flags = 0; 
int add_remove_pt = 0; 
CvPoint pt; 
int i, k, c; 

CvPoint mypoint; 
GLfloat lkPointsX[32]; 
GLfloat lkPointsY[32]; 
// Rotation amounts 
static GLfloat xRot = 0.0f; 
static GLfloat yRot = 0.0f; 
bool boolNightMode; 

//***********************************************************************************************/ 
// GLUT callbacks and functions 

void initGlut(int argc, char **argv); 
void displayFunc(void); 
void idleFunc(void); 
void reshapeFunc(int width, int height); 
void mouseFunc(int button, int state, int x, int y); 
void mouseMotionFunc(int x, int y); 
void keyboardFunc(unsigned char key, int x, int y); 
void specialFunc(int key, int x, int y); 

/***********************************************************************************************/ 

// other [OpenGL] functions 
void countFrames(void); 
void renderBitmapString(float x, float y, float z, void *font, char *string); 
/***********************************************************************************************/ 
bool bFullsreen = false; 
int nWindowID; 
/***********************************************************************************************/ 
// camera attributes 
float viewerPosition[3]    = { 0.0, 0.0, -50.0 }; 
float viewerDirection[3]   = { 0.0, 0.0, 0.0 }; 
float viewerUp[3]          = { 0.0, 1.0, 0.0 }; 

// rotation values for the navigation 
float navigationRotation[3]   = { 0.0, 0.0, 0.0 }; 

// parameters for the navigation 

// position of the mouse when pressed 
int mousePressedX = 0, mousePressedY = 0; 
float lastXOffset = 0.0, lastYOffset = 0.0, lastZOffset = 0.0; 
// mouse button states 
int leftMouseButtonActive = 0, middleMouseButtonActive = 0, rightMouseButtonActive = 0; 
// modifier state 
int shiftActive = 0, altActive = 0, ctrlActive = 0; 
/***********************************************************************************************/ 
// OpenCV variables 
GLuint cameraImageTextureID; 
IplImage  *frame   = 0; // -> image 
int            key = 0; 

bool bInit = false; 
/***********************************************************************************************/ 
void lk_glSpheres() 
{ 
     // drawing lk_points with gl 
     // mypoint.x = points[0][i].x; 
     // mypoint.y = points[1][i].y; 
     
     GLfloat my_lkPointsX; 
     GLfloat my_lkPointsY; 
     GLfloat my_lkPointsZ; 

     printf("CV_X: %d, CV_Y: %d\n",mypoint.x,mypoint.y); 
     
/* 
     if(mypoint.x < 320) mypoint.x = (mypoint.x / 100) * (-1.0f); 
     if(mypoint.x > 320) mypoint.x = (mypoint.x / 100); 
     if(mypoint.y > 160) mypoint.y = (mypoint.y / 100) * (-1.0f); 
     if(mypoint.y < 160) mypoint.y = (mypoint.y / 100);   
*/    
     
     my_lkPointsX = (GLfloat) (mypoint.x / 100.0f); 
     my_lkPointsY = (GLfloat) (mypoint.y / 100.0f); 
     my_lkPointsZ = 20.0f; 
  
     printf("GL_X: %f, GL_Y: %f, GL_Z: %f\n",my_lkPointsX,my_lkPointsY,my_lkPointsZ); 
                    
     // glMatrixMode(GL_MODELVIEW); 
     // glLoadIdentity(); 
     
     // glTranslatef(0.0f,0.0f,0.0f); 
     
     glPushMatrix(); 
        glColor3f(1.0f,0.0f,1.0); 
        glTranslatef(my_lkPointsX,my_lkPointsY,my_lkPointsZ); 
        glutSolidSphere(0.40f,15,15); 
     glPopMatrix(); 
     
     glutSwapBuffers(); 
  
} 
/***********************************************************************************************/ 
void lk(IplImage* frame) 
{ 
     if( !image ) 
{ 
           // allocate all the buffers 
           image = cvCreateImage( cvGetSize(frame), 8, 3 ); 
           cvResize(frame, image); // have to scale to power of two. will work out padding to avoid scale distortion later. 
           image->origin = frame->origin; 
           
         /// -> hier vermutlich ansatz 2.bild zu etablieren 
             // ist in original nicht vorhanden (lkdemo.cpp) 
           trace = cvCreateImage( cvGetSize(frame), 8,3); 
           trace->origin = frame->origin; 
         /// 
           
           grey = cvCreateImage( cvGetSize(frame), 8, 1 ); 
           prev_grey = cvCreateImage( cvGetSize(frame), 8, 1 ); 
           pyramid = cvCreateImage( cvGetSize(frame), 8, 1 ); 
           prev_pyramid = cvCreateImage( cvGetSize(frame), 8, 1 ); 
           points[0] = (CvPoint2D32f*)cvAlloc(MAX_COUNT*sizeof(points[0][0])); 
           points[1] = (CvPoint2D32f*)cvAlloc(MAX_COUNT*sizeof(points[0][0])); 
           status = (char*)cvAlloc(MAX_COUNT); 
           flags = 0; 
} 
     
     // printf("hihanA\n"); 
     cvCopy( frame, image, 0 ); 
     cvCvtColor( image, grey, CV_BGR2GRAY ); 

     if((boolNightMode == true)|| ( night_mode )) 
{ 
          // printf("NIGHTMODE\n"); 
           cvZero( image ); 
} 
     if( need_to_init ) 
{ 
           // automatic initialization 
           IplImage* eig = cvCreateImage( cvGetSize(grey), 32, 1 ); 
           IplImage* temp = cvCreateImage( cvGetSize(grey), 32, 1 ); 
           double quality = 0.01; 
           double min_distance = 10; 
           
           countLK = MAX_COUNT; 
           
           cvGoodFeaturesToTrack( grey, eig, temp, points[1], &countLK, 
                                  quality, min_distance, 0, 3, 1, 0.04 );  // 2.last arg as use_Harris == 1 or 0 
            
           cvFindCornerSubPix( grey, points[1], countLK, 
           cvSize(win_size,win_size), cvSize(-1,-1), 
           cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03)); 
           
           cvReleaseImage( &eig ); 
           cvReleaseImage( &temp ); 
            
           add_remove_pt = 0; 
} 
     else if( countLK > 0 ) 
{ 
           cvCalcOpticalFlowPyrLK( prev_grey, grey, prev_pyramid, pyramid, 
                                   points[0], points[1], countLK, 
                                   cvSize(win_size,win_size), 3, status, 0, 
                                   cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 
                                   20,0.03), flags 
                                 ); 
           
           flags |= CV_LKFLOW_PYR_A_READY; 
            
           for( i = k = 0; i < countLK; i++ ) 
{ 
                 if( add_remove_pt ) 
{ 
                       double dx = pt.x - points[1][i].x; 
                       double dy = pt.y - points[1][i].y; 
                     
                       if( dx*dx + dy*dy <= 25 ) 
{ 
                             add_remove_pt = 0; 
                             continue; 
} 
} 
                
                 if( !status[i] ) 
                    continue; 

                 points[1][k++] = points[1][i]; 
                 
                 // original function 
                 cvCircle( image, cvPointFrom32f(points[1][i]), 3, CV_RGB(0,255,255), -1, 8,0); 
                 
                 // here working with points & testing 
                 //// 
                 // CvPoint mypoint; 
                 mypoint.x = points[0][i].x; 
                 mypoint.y = points[1][i].y; 

                 //lkPointsX[i] = mypoint..x; 
                 //lkPointsY[i] = mypoint..y; 
                 printf("I-loop: %d, X: %d, Y: %d\n",i,mypoint.x,mypoint.y); 
                 
                 cvRectangle( image,cvPoint(mypoint.x,mypoint.y), 
                              cvPoint(mypoint.x+10,mypoint.y+10), 
                              cvScalar(255,0,255) 
                            ); 
                 // here versuch lk_points (myPoints mit glSpheres to draw 
                 // here more spheres better but no images 
                 lk_glSpheres(); 

                 //// 
}  // end for-loop 
            
           countLK = k; 
} 
     
     if( add_remove_pt && countLK < MAX_COUNT ) 
{ 
           points[1][countLK++] = cvPointTo32f(pt); 
           
           cvFindCornerSubPix( grey, points[1] + countLK - 1, 1, 
                               cvSize(win_size,win_size), cvSize(-1,-1), 
                               cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03) 
                             ); 
             
           add_remove_pt = 0; 
} 

     CV_SWAP( prev_grey, grey, swap_temp ); 
     CV_SWAP( prev_pyramid, pyramid, swap_temp ); 
     CV_SWAP( points[0], points[1], swap_points ); 
     need_to_init = 0; 

} 
/***********************************************************************************************/ 
void gltDrawUnitAxes(void) 
{ 
     GLUquadricObj *pObj; // Temporary, used for quadrics 
     
     // Measurements 
     float fAxisRadius = 0.25f; 
     float fAxisHeight = 10.0f; 
     float fArrowRadius = 0.6f; 
     float fArrowHeight = 1.0f; 
     
     // Setup the quadric object 
     pObj = gluNewQuadric(); 
     gluQuadricDrawStyle(pObj, GLU_FILL); 
     gluQuadricNormals(pObj, GLU_SMOOTH); 
     gluQuadricOrientation(pObj, GLU_OUTSIDE); 
     gluQuadricTexture(pObj, GLU_FALSE); 
     
     /////////////////////////////////////////////////////// 
     // Draw the blue Z axis first, with arrowed head 
     glColor3f(0.0f, 0.0f, 1.0f); 
     gluCylinder(pObj, fAxisRadius, fAxisRadius, fAxisHeight, 10, 1); 
     glPushMatrix(); 
        glTranslatef(0.0f, 0.0f, 1.0f); 
        gluCylinder(pObj, fArrowRadius, 0.0f, fArrowHeight, 10, 1); 
        glRotatef(180.0f, 1.0f, 0.0f, 0.0f); 
        gluDisk(pObj, fAxisRadius, fArrowRadius, 10, 1); 
     glPopMatrix(); 
     
     /////////////////////////////////////////////////////// 
     // Draw the Red X axis 2nd, with arrowed head 
     glColor3f(1.0f, 0.0f, 0.0f); 
     glPushMatrix(); 
        glRotatef(90.0f, 0.0f, 1.0f, 0.0f); 
        gluCylinder(pObj, fAxisRadius, fAxisRadius, fAxisHeight, 10, 1); 
        glPushMatrix(); 
           glTranslatef(0.0f, 0.0f, 1.0f); 
           gluCylinder(pObj, fArrowRadius, 0.0f, fArrowHeight, 10, 1); 
           glRotatef(180.0f, 0.0f, 1.0f, 0.0f); 
           gluDisk(pObj, fAxisRadius, fArrowRadius, 10, 1); 
        glPopMatrix(); 
     glPopMatrix(); 
     
     /////////////////////////////////////////////////////// 
     // Draw the Green Y axis 3rd, with arrowed head 
     glColor3f(0.0f, 1.0f, 0.0f); 
     glPushMatrix(); 
        glRotatef(-90.0f, 1.0f, 0.0f, 0.0f); 
        gluCylinder(pObj, fAxisRadius, fAxisRadius, fAxisHeight, 10, 1); 
        glPushMatrix(); 
           glTranslatef(0.0f, 0.0f, 1.0f); 
           gluCylinder(pObj, fArrowRadius, 0.0f, fArrowHeight, 10, 1); 
           glRotatef(180.0f, 1.0f, 0.0f, 0.0f); 
           gluDisk(pObj, fAxisRadius, fArrowRadius, 10, 1); 
        glPopMatrix(); 
     glPopMatrix(); 
     
     //////////////////////////////////////////////////////// 
     // White Sphere at origin 
     glColor3f(1.0f, 1.0f, 1.0f); 
     gluSphere(pObj, 0.05f, 15, 15); 
     
     // Delete the quadric 
     gluDeleteQuadric(pObj); 
} 

/***********************************************************************************************/ 
// new stuff from lk_glDemo.cpp 
void displayFunc(void) 
{ 
     // new stuff 
     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

     IplImage *frame = 0; 

     frame = cvQueryFrame( capture ); 
     // here stuff from lk_glDemo.cpp 
     // lucasKanade-Algo 
     // here, we do the real LK stuff on the image grabbed from the camera 
     lk(frame); 
     
     cvSaveImage( "lk_image.jpg", image ); 
   
     // initialze OpenGL texture    
     glEnable(GL_TEXTURE_RECTANGLE_ARB); 
            
     glGenTextures(1, &cameraImageTextureID); 
     glBindTexture(GL_TEXTURE_RECTANGLE_ARB, cameraImageTextureID); 
           
     glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); 
     glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); 
     glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR); 
     glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_LINEAR); 
           
     glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE); 
           
     if(frame->nChannels == 3) 
{ 
           glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGB, frame->width, frame->height, 0, GL_BGR, GL_UNSIGNED_BYTE, frame->imageData); 
} 
     else if(frame->nChannels == 4) 
{ 
           glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA, frame->width, frame->height, 0, GL_BGRA, GL_UNSIGNED_BYTE, frame->imageData); 
} 
            
      // bInit = true; 
      //  } 

     if(frame) 
{ 
           // clear the buffers 
           glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
           
           glEnable(GL_DEPTH_TEST); 
           glDisable(GL_LIGHTING); 
           glEnable(GL_TEXTURE_RECTANGLE_ARB); 
           
           glMatrixMode(GL_PROJECTION); 
           glLoadIdentity(); 
           
           gluPerspective(50.0, 1.33, 1.0, 100.0); 

           glMatrixMode(GL_MODELVIEW); 
           glLoadIdentity(); 
           
           //  gluLookAt(  viewerPosition[0], viewerPosition[1], viewerPosition[2], 
           //  viewerDirection[0], viewerDirection[1], viewerDirection[2], 
           //  viewerUp[0], viewerUp[1], viewerUp[2]); 
           
           glTranslatef( viewerPosition[0], viewerPosition[1], viewerPosition[2] ); 
           
           // add navigation rotation 
           glRotatef( navigationRotation[0], 1.0f, 0.0f, 0.0f ); 
           glRotatef( navigationRotation[1], 0.0f, 1.0f, 0.0f ); 
           
           // bind texture 
           glBindTexture(GL_TEXTURE_RECTANGLE_ARB, cameraImageTextureID); 
           
           // draw 4 rectangles 
           glBegin(GL_QUADS); 
              glTexCoord2i(0,frame->height); 
              glVertex3f(-15.0,-15.0, 15.0); 
              glTexCoord2i(frame->width,frame->height); 
              glVertex3f(15.0,-15.0, 15.0); 
              glTexCoord2i(frame->width,0); 
              glVertex3f(15.0,15.0, 15.0); 
              glTexCoord2i(0,0); 
              glVertex3f(-15.0,15.0, 15.0); 
           glEnd(); 
           
           glBegin(GL_QUADS); 
              glTexCoord2i(0,frame->height); 
              glVertex3f(15.0,-15.0, -15.0); 
              glTexCoord2i(frame->width,frame->height); 
              glVertex3f(15.0,-15.0, 15.0); 
              glTexCoord2i(frame->width,0); 
              glVertex3f(15.0,15.0, 15.0); 
              glTexCoord2i(0,0); 
              glVertex3f(15.0,15.0, -15.0); 
           glEnd(); 
            
           glBegin(GL_QUADS); 
              glTexCoord2i(0,frame->height); 
              glVertex3f(15.0,-15.0, -15.0); 
              glTexCoord2i(frame->width,frame->height); 
              glVertex3f(-15.0,-15.0, -15.0); 
              glTexCoord2i(frame->width,0); 
              glVertex3f(-15.0,15.0, -15.0); 
              glTexCoord2i(0,0); 
              glVertex3f(15.0,15.0, -15.0); 
           glEnd(); 
           
           glBegin(GL_QUADS); 
              glTexCoord2i(0,frame->height); 
              glVertex3f(-15.0,-15.0, -15.0); 
              glTexCoord2i(frame->width,frame->height); 
              glVertex3f(-15.0,-15.0, 15.0); 
              glTexCoord2i(frame->width,0); 
              glVertex3f(-15.0,15.0, 15.0); 
              glTexCoord2i(0,0); 
              glVertex3f(-15.0,15.0, -15.0); 
           glEnd(); 
  
} // end image 
     
     glDisable(GL_TEXTURE_RECTANGLE_ARB); 
     
     //  old: countFrames(); 


     
     // new stuff 
    // glLoadIdentity(); 
     
     glutSwapBuffers(); 

     // here draw the 3dAxis 
     // from 3dAxis.cpp -> RenderScene() 
     // stopping make texture 

     glDisable(GL_TEXTURE_3D); 

     // render3DAxis 
     // Save the matrix state and do the rotations 
     glPushMatrix(); 
        // Move object back and do in place rotation 
        glTranslatef(0.0f, 0.0f, 20.0f); 
        glRotatef(xRot, 1.0f, 0.0f, 0.0f); 
        glRotatef(yRot, 0.0f, 1.0f, 0.0f); 

        // Draw something 
        gltDrawUnitAxes(); 
        
     // Restore the matrix state 
     glPopMatrix(); 
     
     // Buffer swap 
     glutSwapBuffers();  

     // geht so nicht lk_glSpheres(); 
     lk_glSpheres(); 
} 


/***********************************************************************************************/ 

void idleFunc(void) 
{ 
     glutPostRedisplay(); 
} 

/***********************************************************************************************/ 

void reshapeFunc(int width, int height) 
{ 
     glViewport(0, 0, width, height); 

     /// from lk_glDemo 
     // set projection matrix 
     glMatrixMode(GL_PROJECTION); 
     glLoadIdentity(); 
     gluPerspective(60.0f,(GLfloat)width/(GLfloat)height,0.1f,200.0f); 
     
     // switch back to modelview matrix 
     glMatrixMode(GL_MODELVIEW); 
     glLoadIdentity(); 

} 

/***********************************************************************************************/ 

// mouse callback 
void mouseFunc(int button, int state, int x, int y) 
{ 
     // get the mouse buttons 
     if(button == GLUT_LEFT_BUTTON) 
        if(state == GLUT_DOWN) 
{ 
              leftMouseButtonActive += 1; 
} 
        else 
          leftMouseButtonActive -= 1; 
        else if(button == GLUT_MIDDLE_BUTTON) 
           
           if(state == GLUT_DOWN) 
{ 
                 middleMouseButtonActive += 1; 
                 lastXOffset = 0.0; 
                 lastYOffset = 0.0; 
} 
           else 
              middleMouseButtonActive -= 1; 
           else if(button == GLUT_RIGHT_BUTTON) 
           if(state == GLUT_DOWN) 
{ 
                 rightMouseButtonActive += 1; 
                 lastZOffset = 0.0; 
} 
           else 
              rightMouseButtonActive -= 1; 
              
              mousePressedX = x; 
              mousePressedY = y; 
} 

/***********************************************************************************************/ 

void mouseMotionFunc(int x, int y) 
{ 
     float xOffset = 0.0, yOffset = 0.0, zOffset = 0.0; 
     // navigation 
     
     // rotatation 
     if(leftMouseButtonActive) 
{ 
           navigationRotation[0] += ((mousePressedY - y) * 180.0f) / 200.0f; 
           navigationRotation[1] += ((mousePressedX - x) * 180.0f) / 200.0f; 
           
           mousePressedY = y; 
           mousePressedX = x; 
} 
      
      // panning 
     else if(middleMouseButtonActive) 
{ 
           xOffset = (mousePressedX + x); 
           
           if(!lastXOffset == 0.0) 
{ 
                 viewerPosition[0]  -= (xOffset - lastXOffset) / 8.0; 
                 viewerDirection[0] -= (xOffset - lastXOffset) / 8.0; 
} 
            
           lastXOffset = xOffset; 
            
           yOffset = (mousePressedY + y); 
           
           if(!lastYOffset == 0.0) 
{ 
                 viewerPosition[1]  += (yOffset - lastYOffset) / 8.0; 
                 viewerDirection[1] += (yOffset - lastYOffset) / 8.0;   
} 
            
           lastYOffset = yOffset; 
            
} 
    
     // depth movement 
     else if (rightMouseButtonActive) 
{ 
           zOffset = (mousePressedX + x); 
           
           if(!lastZOffset == 0.0) 
{ 
                 viewerPosition[2] -= (zOffset - lastZOffset) / 5.0; 
                 viewerDirection[2] -= (zOffset - lastZOffset) / 5.0; 
} 
           
           lastZOffset = zOffset; 
} 
} 

/***********************************************************************************************/ 

void keyboardFunc(unsigned char key, int x, int y) 
{ 
     switch(key) 
{ 
           case '\033': // ESC 
              done = true; 
              cvReleaseImage(&image); 
              glDeleteTextures(1, &cameraImageTextureID); 
              printf("Exit with ESC and bye.....\n"); 
              exit(0); 
           break; 
           // special stuff for LK 
  case 'r':  
              printf("Need to Init LK Pressed <r> = Initialitions\n"); 
              need_to_init = 1; 
           break; 
  case 'c': 
              printf("Counts Pressed <c>\n"); 
              counts = 0; 
           break; 
  case 'n': 
              printf("NightMode Pressed <n>\n"); 
              night_mode ^= 1; 
           // end lk-stuff 
           break; 
  case '1':  
              printf("Pressed <1> = Initialitions\n"); //  MakeAxisImages\n"); 
              bool_makePic = false; 
           break; 
  case '2': 
              printf("Pressed <2> = Starting\n"); 
              bool_makePic = true; 
           break; 
  case '3': 
              printf("Pressed <3> = Stopp\n"); 
              bool_makePic = false; 
           break;   
} 
} 
/***********************************************************************************************/ 

void specialFunc(int key, int x, int y) 
{ 
     if(key == GLUT_KEY_UP) 
        xRot-= 5.0f; 
     
     if(key == GLUT_KEY_DOWN) 
        xRot += 5.0f; 
     
     if(key == GLUT_KEY_LEFT) 
        yRot -= 5.0f; 
     
     if(key == GLUT_KEY_RIGHT) 
        yRot += 5.0f; 
                
     xRot = (GLfloat)((const int)xRot % 360); 
     yRot = (GLfloat)((const int)yRot % 360); 

     // Refresh the Window 
     glutPostRedisplay(); 
} 
/***********************************************************************************************/ 

int main(int argc, char **argv) 
{ 
    boolNightMode = false; 

    // initialize camera 
    capture = cvCaptureFromCAM( 0 ); 

    // always check 
    if( !capture ) 
{ 
          printf( "Cannot open initialize WebCam!\n" ); 
          return 1; 
} 

    glutInit(&argc,argv); 
       
    glutInitWindowSize (640, 480); 
    glutInitWindowPosition(100, 100);  

    glutInitDisplayMode ( GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);        
    glutCreateWindow("3D-FaceTracking");     

    glutReshapeFunc(reshapeFunc); 

    glutKeyboardFunc(keyboardFunc); 
    glutSpecialFunc(specialFunc); 
    glutMouseFunc(mouseFunc); 
    glutMotionFunc(mouseMotionFunc); 

// Register callbacks: 
     glutDisplayFunc(displayFunc); 
     glutIdleFunc(displayFunc); 
       
     glutMainLoop(); 

     return (1); 
  
} 
