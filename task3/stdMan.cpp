#include<stdio.h>
#include "cv.h"
#include "highgui.h"
#include<math.h>

using namespace cv;
using namespace std;

// To print imageData values of an image
void PrintImage(IplImage* image)
{
  int j;

 for( j=0 ; j < image->width * image->height * image->nChannels ; j+= image->nChannels)
  {
   
    printf(" B: %10f  G: %10f  R: %10f ", (float) image->imageData[j] , (float) image->imageData[j+1] , (float) image->imageData[j+2]) ;

    if(j%3 == 0);
    printf("\n");
   

  }

}
int main()
{
  
  CvCapture* capture = cvCreateCameraCapture(0);
  cvNamedWindow("frame");
  if(!capture)
    {
      printf("failed");
      return -1;
    }
  
  IplImage* current = cvQueryFrame(capture) ;
  if(!current)
    {
      cout << "failed";
      return -1;
    }
  
  IplImage* sqrMean = cvCreateImage(cvGetSize(current),IPL_DEPTH_64F,3);
  // image for Sum of Xi^2 ==>  (  E(Xi)^2  )
  
  
  IplImage* meanSqr = cvCreateImage(cvGetSize(current),IPL_DEPTH_64F,3);
  // image for ( Sum of Xi )^2 ==>  (E(Xi))^2 
  
  
  IplImage* stdev = cvCreateImage(cvGetSize(current),IPL_DEPTH_64F,3);
  // image for square root of  (1/N)[ sqrMean - (1/N)*meanSqr] where N = nFrames
 
  IplImage* diff = cvCreateImage(cvGetSize(current),8,3);


  int nFrames=1;
  int j;
  while (nFrames <= 60)
    {

     
      nFrames++;
      // for gettig rid for first few frames which have blank image in them
      if(nFrames <=10)
 	continue;
      
      IplImage* previous = NULL;
      previous = cvCloneImage(current);

      current = cvQueryFrame(capture);
      cvSub(current,previous,diff);
      
      for(int i=0 ; i< current->width * current->height * current->nChannels ; i+= current->nChannels)
	{
	  
	    
	  // sqrMean calculation completes when exit from the while loop

	   sqrMean->imageData[i]   += pow(((int)current->imageData[i]   - (int)previous->imageData[i]),2);
	   sqrMean->imageData[i+1] += pow(((int)current->imageData[i+1] - (int)previous->imageData[i+1]),2);
	   sqrMean->imageData[i+2] += pow(((int)current->imageData[i+2] - (int)previous->imageData[i+2]),2);
	   
	   
	   // calculating sum of difference dpixels
	   meanSqr->imageData[i]   +=  current->imageData[i]  - previous->imageData[i];
	   meanSqr->imageData[i+1] += current->imageData[i+1] - previous->imageData[i+1];
	  meanSqr->imageData[i+2] += current->imageData[i+2] - previous->imageData[i+2];
	  
	  
	}// out of for loop
      
      cvReleaseImage(&previous);
      cvShowImage("frame",current);
      char c = cvWaitKey(33);
      if(c==27)
	break;
      
    }// out of while loop
  // correction the adjustment  
  nFrames -= 10;
  
  
  // calculating square of sum
  
  
  for( j=0 ; j < meanSqr->width * meanSqr->height * meanSqr->nChannels ; j+= meanSqr->nChannels)
    {
      meanSqr->imageData[j]   = pow(meanSqr->imageData[j],2)   / (float) nFrames;
      meanSqr->imageData[j+1] = pow(meanSqr->imageData[j+1],2) / (float) nFrames;
      meanSqr->imageData[j+2] = pow(meanSqr->imageData[j+2],2) / (float) nFrames;
      
      
      stdev->imageData[j]   = char (( (float)sqrMean->imageData[j]   - (float) meanSqr->imageData[j])  / nFrames);
      stdev->imageData[j+1] = char (( (float)sqrMean->imageData[j+1] - (float)meanSqr->imageData[j+1]) / nFrames);
      stdev->imageData[j+2] = char (( (float)sqrMean->imageData[j+2] - (float)meanSqr->imageData[j+2]) / nFrames);
      
      
    }


  // PrintImage(stdev);
  //  PrintImage(meanSqr);
  PrintImage(sqrMean);
  cvShowImage("frame",stdev);
  
  
  
  // releasing data
  cvReleaseCapture(&capture);
  cvDestroyWindow("frame");
  cvReleaseImage(&current);
  cvReleaseImage(&sqrMean);
  cvReleaseImage(&meanSqr);
  cvReleaseImage(&stdev);

  return 1;

}
