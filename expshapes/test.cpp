#include "cv.h"
#include "highgui.h"

using namespace cv;

int main()
{
  CvCapture* capture = cvCaptureFromCAM(0);
  IplImage* img= NULL;
  while(1)
    {
      img = cvQueryFrame(capture);
      cvShowImage("naren",img);

      char c = cvWaitKey(30)
	if(c==27)
	  break;

    }

  return 1;
}
