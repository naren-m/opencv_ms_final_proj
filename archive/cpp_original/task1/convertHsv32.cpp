#include <cv.h>
#include "highgui.h"

using namespace std;
using namespace cv;

int main()
{
  
  CvCapture* capture = cvCaptureFromCAM(0);
  if(!capture)
    {
      printf("couldnot initialize");
      return -1;
    }
  
  cvNamedWindow("Sue");
  cvNamedWindow("frame");
  cvNamedWindow("Value");
  cvNamedWindow("imgHsv"); 
  cvNamedWindow("imgHue");
  
  while(1)
    {
      IplImage* frame =0;
      frame = cvQueryFrame(capture);
      if(!frame)
	{ 
	  printf("error"); 
	  break;
	}
      
      IplImage* imgHsv =cvCreateImage(cvGetSize(frame),32,3);
      IplImage* imgSue = cvCreateImage(cvGetSize(frame),32,1);
      IplImage* imgValue = cvCreateImage(cvGetSize(frame),32,1);
      IplImage* imgHue = cvCreateImage(cvGetSize(frame),32,1);
      IplImage* imgRGB = cvCreateImage(cvGetSize(frame),32,3);
      
      cvConvertScale(frame,imgRGB,1/255.);
      cvCvtColor(imgRGB,imgHsv,CV_BGR2HSV);
      
      cvSetImageCOI(imgHsv,1);
      cvCopy(imgHsv,imgHue,NULL);
      
      cvSetImageCOI(imgHsv,2);
      cvCopy(imgHsv,imgSue,NULL);
      
      cvSetImageCOI(imgHsv,3);
      cvCopy(imgHsv,imgValue,NULL);
      
      
      cvShowImage("Sue",imgSue);
      cvShowImage("Value",imgValue);
      cvShowImage("imgHsv",imgHsv);
 cvShowImage("imgHue",imgHue);
      cvShowImage("frame",frame);
      
      cvReleaseImage(&imgHsv);
      cvReleaseImage(&imgSue);
      cvReleaseImage(&imgValue);
      cvReleaseImage(&imgRGB);
      cvReleaseImage(&imgHue);
      
      char c = cvWaitKey(33);
      if(c == 27)
	break;
    }
  cvReleaseCapture(&capture);
  return 0;
}


	








