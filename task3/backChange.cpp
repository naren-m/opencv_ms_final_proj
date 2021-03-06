#include <cv.h>
#include <highgui.h>

using namespace std;
using namespace cv;

// Substracting back ground
// back ground removal Working wiht HSV is not of much help

int main()
{
  
  CvCapture* capture = cvCaptureFromCAM(0);
 
  if(!capture)
    {
      printf("couldnot initialize");
      return -1;
    }
  
  cvNamedWindow("red");  
  cvNamedWindow("green");
  //cvNamedWindow("sub");
  cvNamedWindow("blue");
  cvNamedWindow("final");
  IplImage* background = NULL; 
  IplImage* temp = NULL;
 for(int i=0;i<=30;i++)
    { 
   temp =cvQueryFrame(capture);
 
  background = cvCloneImage(temp);
    }
printf("naren \a");
IplImage* mask=cvCreateImage(cvGetSize(background),8,1);

    
  while(1)
    {
      IplImage* currentFrame = cvQueryFrame(capture);
      
   
      IplImage* sub = cvCreateImage(cvGetSize(background),8,1);
      IplImage* subRgb = cvCreateImage(cvGetSize(background),8,3);
      cvZero(subRgb);
      IplImage* b_blue = cvCreateImage(cvGetSize(background),8,1);
      IplImage* c_blue = cvCreateImage(cvGetSize(background),8,1);
      
      IplImage* b_green = cvCreateImage(cvGetSize(background),8,1);
      IplImage* c_green = cvCreateImage(cvGetSize(background),8,1);
      
      IplImage* b_red = cvCreateImage(cvGetSize(background),8,1);
      IplImage* c_red = cvCreateImage(cvGetSize(background),8,1);

      IplImage* r_sub = cvCreateImage(cvGetSize(background),8,1);
      IplImage* b_sub = cvCreateImage(cvGetSize(background),8,1);
      IplImage* g_sub = cvCreateImage(cvGetSize(background),8,1);
      
      IplImage* abssub = cvCreateImage(cvGetSize(background),IPL_DEPTH_8U,1); 
      IplImage* bck_gray = cvCreateImage(cvGetSize(background),8,1);
      IplImage* cur_gray = cvCreateImage(cvGetSize(background),8,1);
      
      cvCvtColor(background,bck_gray,CV_BGR2GRAY);
      cvCvtColor(currentFrame,cur_gray,CV_BGR2GRAY); 
     
      //Absolute difference  on gray
  
      cvAbsDiff(cur_gray,bck_gray,abssub);
     
      cvThreshold(abssub,abssub,20,255,CV_THRESH_BINARY_INV);
      // working on gray scale ends here
      // cvCvtColor(background,background,CV_BGR2HSV);
      //cvCvtColor(currentFrame,currentFrame,CV_BGR2HSV);
      
      cvSplit(background,b_blue,b_green,b_red,NULL);
      cvSplit(currentFrame,c_blue,c_green,c_red,NULL);

      cvAbsDiff(b_blue,c_blue,b_sub);
      cvAbsDiff(b_green,c_green,g_sub);
      cvAbsDiff(b_red,c_red,r_sub);
    

      cvThreshold(g_sub,g_sub,10,255,CV_THRESH_BINARY_INV);
      cvThreshold(r_sub,r_sub,10,255,CV_THRESH_BINARY_INV);
      cvThreshold(b_sub,b_sub,10,255,CV_THRESH_BINARY_INV);
      /* cvSub(r_sub,abssub,r_sub);
	 cvSub(b_sub,abssub,b_sub);
	 
	 cvSub(g_sub,abssub,g_sub);*/
      
     
 
    
     
      cvAdd(b_sub,g_sub,sub);
      cvAdd(sub,r_sub,sub);
    
      /*   //opening
      cvErode(sub,sub,NULL,2);
      cvDilate(sub,sub,NULL,2);
     
 cvErode(sub,sub,NULL,1);
      cvDilate(sub,sub,NULL,1);

       // closing
      cvDilate(sub,sub,NULL,1);
      cvErode(sub,sub,NULL,1);
     


 // experimenting

      /*    cvDilate(sub,sub,NULL,1);
      cvErode(sub,sub,NULL,1);
      
      cvDilate(sub,sub,NULL,1);
      cvErode(sub,sub,NULL,1);

cvDilate(sub,sub,NULL,1);
      cvErode(sub,sub,NULL,1);
      

cvDilate(sub,sub,NULL,1);
      cvErode(sub,sub,NULL,1);
      
cvDilate(sub,sub,NULL,1);
      cvErode(sub,sub,NULL,1);
      

      // closing
      cvDilate(sub,sub,NULL,1);
      cvErode(sub,sub,NULL,1);

      //experimenting ends here
      
      */
      
  cvCopy(currentFrame,subRgb,sub); 
  
  cvShowImage("final",sub);
  cvShowImage("blue",b_sub);
  cvShowImage("green",g_sub);
  cvShowImage("red",r_sub);
  cvShowImage("frame",subRgb);
  cvShowImage("current",currentFrame);
  cvShowImage("background",background);
  //  cvShowImage("gray_dif",abssub);
  
  cvReleaseImage(&sub);
  cvReleaseImage(&abssub);
  
      cvReleaseImage(&b_blue);
      cvReleaseImage(&c_blue);
      
      cvReleaseImage(&b_green);
      cvReleaseImage(&c_green);
      
      cvReleaseImage(&b_red);
      cvReleaseImage(&c_red);
      
      cvReleaseImage(&b_sub);
      cvReleaseImage(&g_sub);
      cvReleaseImage(&r_sub);

      cvReleaseImage(&cur_gray);
      cvReleaseImage(&bck_gray);
      
      cvReleaseImage(&subRgb);
      
      char c = cvWaitKey(33);
      if(c == 27)
	{
	  //cvReleaseImage(&currentFrame);
	  break;
	}
    }
 
  cvReleaseCapture(&capture);
  cvReleaseImage(&background);
  cvReleaseImage(&mask);
  cvReleaseImage(&temp);
  
  cvDestroyWindow("frame");
  cvDestroyWindow("frame2");
  cvDestroyWindow("abssub");
  cvDestroyWindow("final");
  
  
  return 1;
  
}






  
