#include<stdio.h>
#include "cv.h"
#include "highgui.h"
#include<math.h>
#include<sstream>
using namespace cv;
using namespace std;

// Std Dev wth arrays 
// Having segmentation problem when another array is used




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

  
 
 
  IplImage* diff = cvCreateImage(cvGetSize(current),8,3);


  int nFrames=1;
  int j;

  // image for Sum of Xi^2 ==>  (  E(Xi)^2  )
  double *sqrMeanArray;

// image for ( Sum of Xi )^2 ==>  (E(Xi))^2 
  double *meanSqrArray;

  double *stdevArray;

  stdevArray = new double [10000000];
  meanSqrArray = new double [10000000];
  sqrMeanArray = new double [10000000];


   double var1,var2;

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
      // Reading data from image

         for(int i=0 ; i< current->width * current->height * current->nChannels ; i+= current->nChannels)
    {
      sqrMeanArray[i]=0;
      // meanSqrArray[i]=0;

      //   stdevArray[i]=0;

      }
     
 for(int i=0 ; i< current->width * current->height * current->nChannels ; i+= current->nChannels)
	{
	  
	    
	  // sqrMean calculation completes when exit from the while loop

	  
	   var1 = current->imageData[i] - '0'; 
	   var2 = previous->imageData[i] - '0';
	   sqrMeanArray[i] += pow(var1-var2,2);
	     meanSqrArray[i] +=  var2-var1;

	    var1 = current->imageData[i+1] - '0'; 
	   var2 = previous->imageData[i+1] - '0';
	   sqrMeanArray[i+1] +=pow(var1-var2,2);
 meanSqrArray[i+1] +=  var2-var1;

   var1 = current->imageData[i+2] - '0'; 
	   var2 = previous->imageData[i+2] - '0';
	   sqrMeanArray[i+2] += pow(var1-var2,2);
 meanSqrArray[i+2] +=  var2-var1;
	  /*	
	
	  // calculating sum of difference dpixels
  meanSqrArray[i]   += (double) current->imageData[i]  - (double)previous->imageData[i];
  meanSqrArray[i+1] += (double)current->imageData[i+1] - (double)previous->imageData[i+1];
  meanSqrArray[i+2] += (double) current->imageData[i+2] - (double)previous->imageData[i+2];
	  */
	   /*	    var1 = current->imageData[i] - '0'; 
	   var2 = previous->imageData[i] - '0';
	    meanSqrArray[i] += var1-var2;
	      /*
   var1 = current->imageData[i+1] - '0'; 
	   var2 = previous->imageData[i+1] - '0';
 	  meanSqrArray[i+1] += var1-var2;

   var1 = current->imageData[i+2] - '0'; 
	   var2 = previous->imageData[i+2] - '0';
	   meanSqrArray[i+2] += var1-var2;*/
	  
	  
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
  
  int i;
  for( i=0 ; i< current->width * current->height * current->nChannels ; i+= current->nChannels)
    {
      meanSqrArray[j]   = pow(meanSqrArray[j],2)   /  nFrames;
      meanSqrArray[j+1] = pow(meanSqrArray[j+1],2) /  nFrames;
      meanSqrArray[j+2] = pow(meanSqrArray[j+2],2) / nFrames;
      
      
      stdevArray[j]   =  (sqrMeanArray[j]   -  meanSqrArray[j])  / nFrames;
      stdevArray[j+1] =  (( sqrMeanArray[j+1] - meanSqrArray[j+1]) / nFrames);
      stdevArray[j+2] =  (( sqrMeanArray[j+2] - meanSqrArray[j+2]) / nFrames);
      

 
   
  printf(" B: %lf  G: %lf  R: %lf ", stdevArray[j] , stdevArray[j+1] , stdevArray[j+2]) ;

   if(j%3 == 0);
 printf("\n");  
      
    }
  
  
  
  
  // releasing data
  cvReleaseCapture(&capture);
  cvDestroyWindow("frame");
  cvReleaseImage(&current);
  
  return 1;

}
