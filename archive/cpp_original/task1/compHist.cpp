#include <cv.h>
#include "highgui.h"
#include "functions.h"

using namespace std;
using namespace cv;

int main()
{
  
  CvCapture* capture= cvCaptureFromCAM(0);
  double corel =123456.123;
  
  int h_bins = 30, s_bins = 32;
  int hist_size[] = { h_bins, s_bins };
  float h_ranges[] = { 0, 180 };
  float s_ranges[] = { 0, 255 };
  float* ranges[]  = { h_ranges, s_ranges };
  CvHistogram* hist1 = cvCreateHist( 2, hist_size,CV_HIST_ARRAY, ranges, 1  );
  CvHistogram* hist2 = cvCreateHist( 2, hist_size,CV_HIST_ARRAY, ranges, 1  ); 
  
  if(!capture)
   {
     printf("could not initilize");
     return -1;
   }
  
  while(1)
    {
      IplImage* src1 = cvQueryFrame(capture);  
      IplImage*  hist_img1 = drawHistogram(src1,hist1,h_bins,s_bins);
      //cvWaitKey(1000);
      IplImage* src2 = cvQueryFrame(capture);  
      IplImage* hist_img2 = drawHistogram(src2,hist2,h_bins,s_bins);
      corel = cvCompareHist(hist1,hist2,CV_COMP_CORREL);
      printf("Cv corell-- %d\n",corel);
      
      cvNamedWindow( "Source1", 1 );
      cvShowImage( "Source1", src1 );
      cvNamedWindow( "H-S Histogram1", 1 );
      cvShowImage("H-S Histogram1", hist_img1 );
      
      cvNamedWindow( "Source2", 1 );
      cvShowImage( "Source2", src2 );
      cvNamedWindow( "H-S Histogram2", 1 );
      cvShowImage("H-S Histogram2", hist_img2 );
      
      
      char c = cvWaitKey(33);
      if(c == 27)
       	break; 
    }
 cvReleaseCapture(&capture);
 return 0;
}
