#include <cv.h>
#include <highgui.h>

using namespace std;
using namespace cv;

// Substracting two sequential frames

int main()
{
  
  CvCapture* capture = cvCaptureFromCAM(0);
  
  if(!capture)
    {
      printf("couldnot initialize");
      return -1;
    }
  
  cvNamedWindow("frame");  
  cvNamedWindow("frame2");
  cvNamedWindow("sub");
  cvNamedWindow("abssub");
    
 
  int count=0;

  
  while(1)
    {
      IplImage* curr1 =cvQueryFrame(capture);
      IplImage* prev1 =cvCreateImage(cvGetSize(curr1),8,3);
      
  while(count<=2)
    {
      // curr1=cvQueryFrame(capture);
      count++;
      if(count==1)
	prev1=cvCloneImage(curr1);
    }
  count=0;
  curr1=cvQueryFrame(capture);

      IplImage* imgRGB = cvCreateImage(cvGetSize(curr1),8,3);
      IplImage* sub = cvCreateImage(cvGetSize(curr1),8,3);
      
      IplImage* abssub = cvCreateImage(cvGetSize(curr1),IPL_DEPTH_8U,3);    
            
      cvSub(curr1,prev1,sub,NULL);
      cvAbsDiff(curr1,prev1,abssub);     
      // cvZero(prev1);
           
      cvShowImage("sub",sub);
      cvShowImage("abssub",abssub);
      cvShowImage("frame",curr1);
      cvShowImage("frame2",prev1);
      
      cvReleaseImage(&prev1);
      cvReleaseImage(&imgRGB);
     
      cvReleaseImage(&abssub);
      cvReleaseImage(&sub);
     
      
char c = cvWaitKey(33);
 if(c == 27)
	  break;
    }
  cvReleaseCapture(&capture);

}






  
