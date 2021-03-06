#include "cv.h"
#include "highgui.h"

using namespace std;


IplImage* GetThresholdedImage(IplImage* img)
{
  IplImage* imgHSV =cvCreateImage(cvGetSize(img) ,8,3);
  cvCvtColor(img ,imgHSV,CV_BGR2HSV);
  
   IplImage* imgThreshed = cvCreateImage(cvGetSize(img),8,1);
   cvInRangeS(imgHSV,cvScalar(110 ,100, 30),cvScalar(130,255,255),imgThreshed);
  cvReleaseImage(&imgHSV);
  return imgHSV;
  
}

int main()
{
  
  CvCapture* capture = cvCaptureFromCAM(0);
  if(!capture)
    {
      printf("couldnot initialize");
      return -1;
    }
  
  
  
  
  //cvNamedWindow("video");
  cvNamedWindow("thresh");
  
  IplImage* imgScribble = NULL;
  
  while(1)
    {
      IplImage* frame =0;
      frame = cvQueryFrame(capture);
      if(!frame)
	{ printf("error"); break;}
         if(imgScribble == NULL)
	   { imgScribble = cvCreateImage(cvGetSize(frame) , 8,3);	}
            IplImage* imgYellowThresh = GetThresholdedImage(frame);
	     CvMoments *moments = (CvMoments*)malloc(sizeof(CvMoments));
      cvMoments(imgYellowThresh,moments,1);

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

  
    if(lastX>0 && lastY>0 && posX>0 && posY>0)
	{
	  cvLine(imgScribble, cvPoint(posX , posY) , cvPoint(lastX,lastY) ,cvScalar(0,255,255),5 );
	  }
      cvNamedWindow("scribble");	 
        cvAdd(frame , imgScribble ,frame);
    cvShowImage("thresh",imgYellowThresh);
      cvShowImage("video",frame);
      cvShowImage("scribble",imgScribble);
	  char c = cvWaitKey(33);
	  if(c == 27)
	    break;
	  cvReleaseImage(&imgYellowThresh);
	   delete moments;
    }
  cvReleaseCapture(&capture);
  return 0;
}


	








