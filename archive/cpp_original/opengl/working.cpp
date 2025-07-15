#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <iostream>
using namespace std;

CvFont font;

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
	 CvPoint *pt[4];
            for(int i=0;i<4;i++)
	      pt[i] = (CvPoint*)cvGetSeqElem(result, i);
	    
	    	cvLine(ret, *pt[0], *pt[1], cvScalar(255),3);
		cvLine(ret, *pt[1], *pt[2], cvScalar(255),3);
		cvLine(ret, *pt[2], *pt[3], cvScalar(255),3);
		cvLine(ret, *pt[3], *pt[0], cvScalar(255),3);
	    // cvFillConvexPoly(ret,*pt,4,cvScalar(255),8,0);
		//	printf("%s",*pt[3]);
       }
     
     contours = contours->h_next;
   }
 cvReleaseImage(&temp);
 cvReleaseMemStorage(&storage);
	return ret;
}


int main()
{
  int i;
  CvCapture *capture = cvCaptureFromCAM(0);
  IplImage* img ;//= cvLoadImage("1.png");
  IplImage* onImage = cvLoadImage("1.png");
  cvShowImage("on image" ,onImage);

  while (1)
      {
	img = cvQueryFrame(capture);
	IplImage* contourDrawn = 0;
	cvNamedWindow("original");
	//	cvShowImage("naren",img);
	contourDrawn = DetectAndDrawQuads(img);
   
	cvAdd(contourDrawn ,img ,img);


	CvPoint centre = cvPoint(moments->m10/moments->m00, moments->m01/moments->m00);
	
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0, 1, CV_AA);
	
	cvPutText(img, "Rectangle", /*cvPoint(posX,posY)*/ centre, &font, cvScalar(0, 255, 0, 0));
	

	
	cvShowImage("original", img);
	cvNamedWindow("contours");
	
	cvShowImage("contours", contourDrawn);
	
	char c;
	c=cvWaitKey(30);
	if(c==27)
      break;
      }
  return 1;
}
