#include <cv.h>
#include <highgui.h>

using namespace std;
using namespace cv;

// Substracting back ground

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
  //cvNamedWindow("sub");
  cvNamedWindow("abssub");
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
      IplImage* bck_gray = cvCreateImage(cvGetSize(background),8,1);
      IplImage* cur_gray = cvCreateImage(cvGetSize(background),8,1);
      IplImage* abssub = cvCreateImage(cvGetSize(background),IPL_DEPTH_8U,1); 
     
      cvZero(sub);
      cvZero(bck_gray);
      cvZero(cur_gray);

      cvCvtColor(background,bck_gray,CV_BGR2GRAY);
      cvCvtColor(currentFrame,cur_gray,CV_BGR2GRAY);      

      //Absolute difference    
      cvAbsDiff(cur_gray,bck_gray,abssub);
     
      cvThreshold(abssub,abssub,20,255,CV_THRESH_BINARY);
      
      //v cvErode(abssub,abssub,0,3);
      //  cvSmooth(abssub,abssub,CV_GAUSSIAN,3,3);
      cvDilate(abssub,abssub,0,1);


      //Difference
    
      cvSub(bck_gray,cur_gray,sub,NULL);     
      
      cvThreshold(sub,sub,20 ,255,CV_THRESH_BINARY);
  
      //  cvSub(bck_gray,cur_gray,sub,sub);
    

      //  cvThreshold(sub,sub,10,255,CV_THRESH_BINARY);
      //  cvSmooth(sub,sub,CV_GAUSSIAN,3,3);
      //   cvDilate(sub,sub,NULL,1);
      
      
      // cvAdd(sub,abssub,sub);  
      //works pretty well when we are closer to  camera..
      
      cvCopy(currentFrame,subRgb,sub);     
      // cvSub(abssub,sub,sub);
      
      cvShowImage("abssub",abssub);
      cvShowImage("frame",bck_gray);
      cvShowImage("frame2",sub);
      cvShowImage("final",subRgb);
      
      cvReleaseImage(&sub);
      cvReleaseImage(&abssub);
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






  
