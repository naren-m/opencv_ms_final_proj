#include "highgui.h"
#include<stdio.h>
#include<cv.h>

using namespace cv;
using namespace std;

int main( int argc, char** argv ) {
  cvNamedWindow( "Example2", CV_WINDOW_AUTOSIZE );
CvCapture* capture = cvCaptureFromFile("test.avi");
 if(!capture){
   printf("couldnot");
   return -1; 
 }
 IplImage* frame;
 while(1) {
   frame = cvQueryFrame( capture );
   if( !frame ) break;
   cvErode(frame,frame,0,2);
   cvShowImage( "Example2", frame );
   char c = cvWaitKey(33);
   if( c == 27 ) break;
 }
 cvReleaseCapture( &capture );
 cvDestroyWindow( "Example2" );
}
