
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
GLuint onImageTextureID;
IplImage  *frame   = 0; // -> image 
int            key = 0; 

bool bInit = false; 
/// open cv for identifyin rectangular red paper

CvFont font;
CvPoint *onPoint[4];
CvMoments *moments = (CvMoments*)malloc(sizeof(CvMoments));

double angle( CvPoint* pt1, CvPoint* pt2, CvPoint* pt0 )
{
  double dx1 = pt1->x - pt0->x;
  double dy1 = pt1->y - pt0->y;
  double dx2 = pt2->x - pt0->x;
  double dy2 = pt2->y - pt0->y;
  return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

IplImage* DetectAndDrawQuads(IplImage* img)
{
  CvSeq* contours;
  CvSeq* result;
  
  CvMemStorage *storage = cvCreateMemStorage(0);
  
  
 IplImage* imgHSV = cvCreateImage(cvGetSize(img),8,3);
 cvCvtColor(img,imgHSV,CV_BGR2HSV);
 
 
 
 IplImage* imgThreshed = cvCreateImage(cvGetSize(img),8,1);
 cvInRangeS(imgHSV,cvScalar(160,50,50),cvScalar(180,255,255),imgThreshed);
 

 cvMoments(imgThreshed,moments,1);
 /*
 //opening
 cvErode(imgThreshed,imgThreshed,NULL,2);
 cvDilate(imgThreshed,imgThreshed,NULL,2);
  

 //closing
 cvDilate(imgThreshed,imgThreshed,NULL,1);
 cvErode(imgThreshed,imgThreshed,NULL,1);
 */


 cvShowImage("threshed",imgThreshed);
 
 
 IplImage* ret = cvCreateImage(cvGetSize(img), 8, 3);
 IplImage* temp = cvCreateImage(cvGetSize(img), 8, 1);
 
 cvCvtColor(img, temp, CV_BGR2GRAY);
 
 cvFindContours(imgThreshed, storage, &contours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));
 
 while(contours)
   {
     result = cvApproxPoly(contours, sizeof(CvContour), storage, CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0);
     
     if(result->total==4 && fabs(cvContourArea(result, CV_WHOLE_SEQ)) > 20)
       {
	 //	 CvPoint *pt[4];
            for(int i=0;i<4;i++)
	     onPoint[i] = (CvPoint*)cvGetSeqElem(result, i);
	    
	    	cvLine(ret, *onPoint[0], *onPoint[1], cvScalar(255),3);
		cvLine(ret, *onPoint[1], *onPoint[2], cvScalar(255),3);
		cvLine(ret, *onPoint[2], *onPoint[3], cvScalar(255),3);
		cvLine(ret, *onPoint[3], *onPoint[0], cvScalar(255),3);
	    // cvFillConvexPoly(ret,*pt,4,cvScalar(255),8,0);
	
       }
     
     contours = contours->h_next;
   }
 // cvReleaseImage(&imgThreshed);
 //cvReleaseImage(&img);
 //cvReleaseImage(&imgHSV);
 cvReleaseImage(&temp);
 cvReleaseMemStorage(&storage);
	return ret;
}


// new stuff from lk_glDemo.cpp 
void displayFunc(void) 
{ 
     // new stuff 
     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

     IplImage *frame = 0; 
     frame = cvQueryFrame( capture ); 
        //  cvSaveImage( "lk_image.jpg", image ); 
        IplImage* contourDrawn = 0;
	contourDrawn = DetectAndDrawQuads(frame);
	cvAdd(contourDrawn ,frame ,frame);
	CvPoint centre = cvPoint(moments->m10/moments->m00, moments->m01/moments->m00);
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0, 1, CV_AA);
	cvPutText(frame, "Rectangle", /*cvPoint(posX,posY)*/ centre, &font, cvScalar(0, 255, 0, 0));
	

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
	       /*float width = (float) frame->width;
	     float height = (float) frame->height;          
	    glOrtho(0,width,height,0,2.0,-2.0);
	    glLoadIdentity();*/
           
           //gluPerspective(50.0, 1.33, 1.0, 100.0);
	       //uPerspective(35.0, 1.33, 1.0, 100.0);  

	   glMatrixMode(GL_MODELVIEW); 
           glLoadIdentity(); 
           
          
           
	   /*glTranslatef( viewerPosition[0], viewerPosition[1], viewerPosition[2] ); 
           
           // add navigation rotation 
	       glRotatef( navigationRotation[0], 1.0f, 0.0f, 0.0f ); 
	    glRotatef( navigationRotation[1], 0.0f, 1.0f, 0.0f ); 
	   */
           // bind texture 
           glBindTexture(GL_TEXTURE_RECTANGLE_ARB, cameraImageTextureID); 
           
           // draw 4 rectangles 
	      glBegin(GL_QUADS); 
              glTexCoord2i(0,frame->height); 
              glVertex2f(0.0,0.0); 
              glTexCoord2i(frame->width,frame->height); 
              glVertex2f(640.0,0.0); 
              glTexCoord2i(frame->width,0); 
              glVertex2f(640.0,480.0); 
              glTexCoord2i(0,0); 
              glVertex2f(0.0 , 480.0); 
           glEnd(); 
           
         
  

	   	   	   // code for on red paper image
	   	   IplImage* onImage = cvLoadImage("1.png");

	   if(onImage){
	     //  printf("error");

	  

	   // cvShowImage("naren",onImage);

   // initialze OpenGL texture    
	    glEnable(GL_TEXTURE_RECTANGLE_ARB); 
            
	    /*glGenTextures(1, &onImageTextureID); 
     glBindTexture(GL_TEXTURE_RECTANGLE_ARB, onImageTextureID); 
           
     glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); 
     glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); 
     glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR); 
     glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_LINEAR); 
     
     glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL); 
     
     if(onImage->nChannels == 3) 
       { 
	 glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGB, onImage->width, onImage->height, 0, GL_BGR, GL_UNSIGNED_BYTE, onImage->imageData); 
       } 
     else if(onImage->nChannels == 4) 
       { 
	 glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA, onImage->width, onImage->height, 0, GL_BGRA, GL_UNSIGNED_BYTE, onImage->imageData); 
       } 
     // testing
     
     //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
     
     glEnable(GL_DEPTH_TEST); 
     glDisable(GL_LIGHTING); 
     glEnable(GL_TEXTURE_RECTANGLE_ARB); 
     
     //glMatrixMode(GL_PROJECTION); 
     float width = (float) onImage->width;
     float height = (float) onImage->height;          
     glOrtho(0,width,height,0,2.0,-2.0);
     glLoadIdentity(); 
     
     //gluPerspective(50.0, 1.33, 1.0, 100.0);
     //gluPerspective(35.0, 1.33, 1.0, 100.0);  

     glMatrixMode(GL_MODELVIEW); 
     glLoadIdentity(); 
     
     
     glBindTexture(GL_TEXTURE_RECTANGLE_ARB, onImageTextureID); 
     
     
     glBegin(GL_QUADS); 
     
     glTexCoord2i(0,0);     
     //glTexCoord2i(0,onImage->height);
     glVertex3f(-10.0,-10.0, 15.0); 
      // glVertex3f(onPoint[0]->x,onPoint[0]->y, 15.0); 
      

     glTexCoord2i(0,1);   
     glTexCoord2i(onImage->width,onImage->height);
     glVertex3f(10.0,-10.0, 15.0);  
     // glVertex3f(onPoint[1]->x,onPoint[1]->y, 15.0); 
     
     glTexCoord2i(1,1);   
     glTexCoord2i(onImage->width,0); 
     glVertex3f(10.0,10.0, 15.0); 
     // glVertex3f(onPoint[2]->x,onPoint[2]->y, 15.0); 
     
      
     glTexCoord2i(1,0); 
     glVertex3f(-10.0,10.0, 15.0); 
     //glVertex3f(onPoint[3]->x,onPoint[3]->y, 15.0); 

     
     glEnd(); 
	      
     printf("point 1  x->%f , y->%f\n ",onPoint[0]->x,onPoint[0]->y);
  printf("point 2  x->%f , y->%f\n",onPoint[1]->x,onPoint[1]->y);
  printf("point 3  x->%f , y->%f\n ",onPoint[2]->x,onPoint[2]->y);
  printf("point 4  x->%f , y->%f\n ",onPoint[3]->x,onPoint[3]->y);

	    */
     
     
     
	   }// end of onPoint if


	   
	   else {
	     printf("error");
	     exit(0);
	   }
	   
	   
	   
 } // end image 

glDisable(GL_TEXTURE_RECTANGLE_ARB); 
  CvCapture *capture ;

     //  old: countFrames(); 



     // new stuff 
// glLoadIdentity(); 

glutSwapBuffers(); 



glDisable(GL_TEXTURE_3D); 

// render3DAxis 
// Save the matrix state and do the rotations 
glPushMatrix(); 
// Move object back and do in place rotation 
//Translatef(0.0f, 0.0f, 20.0f); 
//Rotatef(xRot, 1.0f, 0.0f, 0.0f); 
//Rotatef(yRot, 0.0f, 1.0f, 0.0f); 


     
     // Buffer swap 
     glutSwapBuffers();  

    
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
