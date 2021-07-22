#include "cv.h"
#include "highgui.h"

using namespace cv;

IplImage* GetYellowThresholdedImage(IplImage* img)
{
  IplImage* imgHSV =cvCreateImage(cvGetSize(img) ,8,3);
  cvCvtColor(img ,imgHSV,CV_BGR2HSV);
  
  IplImage* imgThreshed = cvCreateImage(cvGetSize(img),8,1);
  cvInRangeS(imgHSV, cvScalar(20 ,100, 100),cvScalar(30,255,255),imgThreshed);
  cvReleaseImage(&imgHSV);
  return imgThreshed;
  
}
IplImage* GetBlueThresholdedImage(IplImage* img)
{
  IplImage* imgHSV =cvCreateImage(cvGetSize(img) ,8,3);
  cvCvtColor(img ,imgHSV,CV_BGR2HSV);
  
  IplImage* imgThreshed = cvCreateImage(cvGetSize(img),8,1);
  cvInRangeS(imgHSV, cvScalar(100 ,100, 100),cvScalar(180,255,255),imgThreshed);
  cvReleaseImage(&imgHSV);
  return imgThreshed;
  
}

IplImage* GetScribbledImage(IplImage* img , IplImage* imgScribble)
{  
  
  
  CvMoments *moments = (CvMoments*)malloc(sizeof(CvMoments));
  cvMoments(img,moments,1);
  
      double moment10 = cvGetSpatialMoment(moments, 1,0);
      double moment01 = cvGetSpatialMoment(moments,0 ,1);
      double area = cvGetCentralMoment(moments,0,0);
      
      static int posX = 0;
      static int posY = 0;
      
      int lastX = posX;
      int lastY = posY;
      
      posX = moment10/area;
      posY = moment01/area;
      
      printf("position (%d , %d)\n", posX, posY);
      
      
      if( posX>0 && posY>0)
	{
	  cvRectangle(imgScribble, cvPoint(posX , posY) , cvPoint(posX+10 , posY+10) ,cvScalar(0,255,255),1,0,0 );
	}
      delete moments;
      return imgScribble;
      
}


int main()
{
  
  CvCapture* capture = cvCaptureFromCAM(0);
  if(!capture)
    {
      printf("couldnot initialize");
      return -1;
    }
  
  
  
  
  cvNamedWindow("video");
  //cvNamedWindow("Yellow thresh");
  cvNamedWindow("Blue Thresh");
  
  IplImage* imgScribbleYellow = NULL;
  IplImage* imgScribbleBlue = NULL;
  
  while(1)
    {
      IplImage* frame =0;
      frame = cvQueryFrame(capture);
      if(!frame)
	{ printf("error"); break;
	}
      
 if(imgScribbleYellow == NULL)
   {
     imgScribbleYellow = cvCreateImage(cvGetSize(frame) , 8,3);	
   }
 
 if(imgScribbleBlue == NULL)
   {
	  imgScribbleBlue = cvCreateImage(cvGetSize(frame) , 8,3);	
   }
 
 
 
 // IplImage* imgThreshYellow = GetYellowThresholdedImage(frame);
 IplImage* imgThreshBlue = GetBlueThresholdedImage(frame);
	    
 //imgScribbleYellow = GetScribbledImage(imgThreshYellow,imgScribbleYellow);
 imgScribbleBlue = GetScribbledImage(imgThreshBlue,imgScribbleBlue);
 
 cvAdd(frame , imgScribbleBlue,frame);
 // cvNamedWindow("scribble Blue");
 // cvShowImage("Yellow thresh",imgThreshYellow);
 cvShowImage("video",frame);
 cvShowImage("Blue Thresh",imgThreshBlue);
 //    cvShowImage("scribble Blue",imgScribbleBlue);
 char c = cvWaitKey(33);
 if(c == 27)
	    break;
 // cvReleaseImage(&imgThreshYellow);
 cvReleaseImage(&imgThreshBlue);
 
    }
  cvReleaseCapture(&capture);
  return 0;
}


	








