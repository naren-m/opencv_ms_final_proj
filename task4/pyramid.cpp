#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#include <stdlib.h>
// Pyramid
using namespace cv;

Mat src, dst,tmp;

int main()
{

  CvCapture* capture = cvCaptureFromCAM(0);



  while(1)
    {
      src = cvQueryFrame(capture);
          
      pyrDown(src,dst,Size(src.cols/2 ,src.rows/2));
      
      imshow("pyrDown",dst);
      imshow("video",src);
      int c = waitKey(10);
      
      if((char) c == 27)
	break;
      
    }


}
