#include "cv.h"
#include "highgui.h"
#include <iostream>
#include <math.h>
#include <string.h>
using namespace std;
using namespace cv;

int N = 2,thresh =50;
double angle( Point pt1, Point pt2, Point pt0 )
{
  double dx1 = pt1.x - pt0.x;
  double dy1 = pt1.y - pt0.y;
  double dx2 = pt2.x - pt0.x;
  double dy2 = pt2.y - pt0.y;
  return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

int main( int argc, const char** argv )
{
  CvCapture* capture;
  Mat frame,linesframe,originalframe;
  Mat dst,cdst,contourcan;
  //vector<vector<Point> > contours;
  vector<vector<Point> > squares;
  // cdst =NULL;
  
  capture = cvCaptureFromCAM(0);
  
     while( true )
       {
	originalframe = cvQueryFrame( capture );

	frame = originalframe.clone();
	 Canny(frame,dst,50,150,3);
	 cvtColor(dst,cdst, CV_GRAY2BGR);
	 Canny(frame ,contourcan,50,150,3);
   
	 // 
	 
	 linesframe = frame.clone();
	 
	 
	 
	 /// checkin delete after working
	 
	 
 
	 vector<Vec4i> lines;

	 HoughLinesP(dst,lines,/*resolution of param 'r' */1 ,/*resolution param theta*/CV_PI/180   ,/*minimum no. of intersections */70   ,/* min no. of points that can form a line*/50    ,/*max gap b/w 2 points to be considered in same line */30);
	 
	 for( size_t i=0; i<lines.size();i++)
	   {
	     Vec4i l = lines[i];
	     line(cdst , Point(l[0],l[1]) , Point(l[2],l[3]) , Scalar(0,0255) , 3, CV_AA);
	     
	     
	     line(linesframe ,Point(l[0],l[1]) , Point(l[2],l[3]) , Scalar(0,0255) , 3, CV_AA);
	     
	   }

   ///
	 
	 	 

	 	 
	 
	 /// code for hough lines
	 // working good
	 

     

   /* vector<Vec2f> lines;
   HoughLines(dst , lines , 1 ,CV_PI/180 , 100 , 0 ,0);

   for( size_t i = 0; i < lines.size(); i++ )
     {
       float rho = lines[i][0], theta = lines[i][1];
       Point pt1, pt2;
       double a = cos(theta), b = sin(theta);
       double x0 = a*rho, y0 = b*rho;
       pt1.x = cvRound(x0 + 1000*(-b));
       pt1.y = cvRound(y0 + 1000*(a));
       pt2.x = cvRound(x0 - 1000*(-b));
       pt2.y = cvRound(y0 - 1000*(a));
       line( cdst, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
       }*/

   imshow("detected lines" , cdst);
   imshow("canny" , dst);
   imshow("lines frame",linesframe);
   // releaseImage(dst);

   char c=waitKey(30);
   if(c ==27)
     break;

     }
   return 0;
 

     }

