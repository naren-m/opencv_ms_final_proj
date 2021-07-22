#include<iostream> 
// openCV 
#include "cv.h" 
                  
#include "highgui.h" 
// openGL & GLUI 
#include <GL/glut.h>   // into usr/include/GL/ 
#include <GL/gl.h> 
#include <GL/freeglut.h>
//#include "glui.h" // <GL/glui.h>   // into usr/include/GL/ 
// c 
#include <math.h> 

// C-Stuff for RS232 
#include <stdio.h> 
#include <stdlib.h> 

//using namespace cv;

 using namespace std; 

GLuint cameraImageTextureID; 
GLuint onImageTextureID;
IplImage *onImage;

void initGlut(int argc, char **argv); 
void displayFunc(IplImage*); 
void idleFunc(void); 
void reshapeFunc(int width, int height); 
void mouseFunc(int button, int state, int x, int y); 
void mouseMotionFunc(int x, int y); 
void keyboardFunc(unsigned char key, int x, int y); 
void specialFunc(int key, int x, int y); 
void displayOnImageFunc(IplImage*);
  CvCapture *capture ;
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
  cvInRangeS(imgHSV,cvScalar(150,100,50),cvScalar(190,255,255),imgThreshed);
  cvMoments(imgThreshed,moments,1);

 //opening
 cvErode(imgThreshed,imgThreshed,NULL,2);
 cvDilate(imgThreshed,imgThreshed,NULL,2);
  

 //closing
 cvDilate(imgThreshed,imgThreshed,NULL,1);
 cvErode(imgThreshed,imgThreshed,NULL,1);

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
	    
	    /*	cvLine(ret, *pt[0], *pt[1], cvScalar(255),3);
		cvLine(ret, *pt[1], *pt[2], cvScalar(255),3);
		cvLine(ret, *pt[2], *pt[3], cvScalar(255),3);
		cvLine(ret, *pt[3], *pt[0], cvScalar(255),3);
	    // cvFillConvexPoly(ret,*pt,4,cvScalar(255),8,0);
	    //	printf("%s",*pt[3]);*/


	    //onPoint[i] = (CvPoint*)cvGetSeqElem(result, i);
	    
	    	cvLine(ret, *onPoint[0], *onPoint[1], cvScalar(255),3);
		cvLine(ret, *onPoint[1], *onPoint[2], cvScalar(255),3);
		cvLine(ret, *onPoint[2], *onPoint[3], cvScalar(255),3);
		cvLine(ret, *onPoint[3], *onPoint[0], cvScalar(255),3);
       }
     
     contours = contours->h_next;
   }
 cvReleaseImage(&temp);
 cvReleaseMemStorage(&storage);
 // cvReleaseImage(&imgHSV);
	return ret;
}

void displayFunc(IplImage *frame)
{
          
     
     if(frame) 
{ 
           // clear the buffers 
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
           
           glEnable(GL_DEPTH_TEST); 
           glDisable(GL_LIGHTING); 
	      glEnable(GL_TEXTURE_RECTANGLE_ARB); 
           
	   glMatrixMode(GL_PROJECTION);
	   glLoadIdentity();
	   //   gluOrtho2D(0,frame->width,frame->height,0);
	     glOrtho(0,frame->width,frame->height,0,1.0,-2.0);
glMatrixMode(GL_MODELVIEW); 
           glLoadIdentity(); 
	   glBindTexture(GL_TEXTURE_RECTANGLE_ARB, cameraImageTextureID); 
	     glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0,  0, 0,  frame->width, frame->height, GL_BGR, GL_UNSIGNED_BYTE, frame->imageData); 
           // draw 4 rectangles 
	   glBegin(GL_QUADS); 
	   glTexCoord2i(0,0);     glVertex3f(0.0,0.0,1.75); 
	   glTexCoord2i(640,0);   glVertex3f(frame->width,0.0,1.75); 
	   glTexCoord2i(640,480); glVertex3f(frame->width,frame->height,1.75); 
	   glTexCoord2i(0,480);   glVertex3f(0.0 , frame->height,1.75); 
	   glEnd(); 

	    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, onImageTextureID);
	   glBegin(GL_QUADS);

	   /*  glTexCoord2i(0,0);                            glVertex3f(0.0,0.0,1.5);           
	   glTexCoord2i(onImage->width,0);              glVertex3f(frame->width/2,0.0,1.5); 
	   glTexCoord2i(onImage->width,onImage->height); glVertex3f(frame->width/2,frame->height/2,1.5);
	   glTexCoord2i(0,onImage->height);               glVertex3f(0.0 , frame->height/2,1.5); */

	   glTexCoord2i(0,0);                            glVertex3f(onPoint[0]->x,onPoint[0]->y,1.5);           
	   glTexCoord2i(onImage->width,0);              glVertex3f(onPoint[1]->x,onPoint[1]->y,1.5); 
	   glTexCoord2i(onImage->width,onImage->height); glVertex3f(onPoint[2]->x,onPoint[2]->y,1.5);
	   glTexCoord2i(0,onImage->height);               glVertex3f(onPoint[3]->x,onPoint[3]->y,1.5); 



	   glEnd();

 
 }
     

glutSwapBuffers(); 

}

void initTexture(IplImage *frame)
{
 // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

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
   
// on Image textyure

 glGenTextures(1, &onImageTextureID); 
     glBindTexture(GL_TEXTURE_RECTANGLE_ARB, onImageTextureID); 
           
     glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); 
     glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); 
     glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR); 
     glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_LINEAR); 
           
     glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE); 

   glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGB, onImage->width, onImage->height, 0, GL_BGR, GL_UNSIGNED_BYTE, onImage->imageData); 


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

void keyboardFunc(unsigned char key, int x, int y) 
{ 
     switch(key) 
{ 
           case '\033': // ESC 
	   
              glDeleteTextures(1, &cameraImageTextureID); 
              printf("Exit with ESC and bye.....\n"); 
              exit(0); 
           break; 
           // special stuff for LK 

} 
} 
/***********************************************************************************************/ 



int main(int argc, char **argv) 
{ 
   glutInit(&argc,argv); 
       
    glutInitWindowSize (640, 480); 
    glutInitWindowPosition(100, 100);  

    glutInitDisplayMode ( GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);        
    glutCreateWindow("3D-FaceTracking");     
 
    // initialize camera 
    capture = cvCaptureFromCAM( 0 ); 
    IplImage *camFrame = 0;;
    onImage = cvLoadImage("1.png");
    // always check 
    if( !capture ) 
{ 
          printf( "Cannot open initialize WebCam!\n" ); 
          return 1; 
} 

    camFrame = cvQueryFrame(capture);
    initTexture(camFrame);

    while(1)
      {

  	camFrame = cvQueryFrame(capture);
	IplImage* contourDrawn = 0;
	cvNamedWindow("original");
	//	cvShowImage("naren",img);
	contourDrawn = DetectAndDrawQuads(camFrame);
   
	cvAdd(contourDrawn ,camFrame ,camFrame);


	CvPoint centre = cvPoint(moments->m10/moments->m00, moments->m01/moments->m00);
	
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0, 1, CV_AA);
	
	cvPutText(camFrame, "Rectangle", /*cvPoint(posX,posY)*/ centre, &font, cvScalar(0, 255, 0, 0));
	
	
	//	cvShowImage("original", camFrame);
	cvNamedWindow("contours");	
	cvShowImage("contours", contourDrawn);
	cvShowImage("onImage" , onImage);
	
	glutMainLoopEvent();
	glutKeyboardFunc(keyboardFunc); 
	displayFunc(camFrame); 
	//	displayOnImageFunc(onImage);

 cvReleaseImage(&contourDrawn);
	int c = cvWaitKey(30);
     if(c==27)
       break;
      }
    cvReleaseCapture(&capture);   
    cvReleaseImage(&onImage);
   
 return (1); 
    
} 
