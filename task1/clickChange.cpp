#include <cv.h>
#include <highgui.h>

using namespace std;
using namespace cv;

CvScalar hsvPix;

void mouse_callback(int event, int x, int y, int flags, void* param );
IplImage* GetThresholdedImage(IplImage* img);
float maximum(int x,int y,int z);
float minimum(int x,int y,int z);

int main()
{
  
  CvCapture* capture = cvCaptureFromCAM(0);
  if(!capture)
    {
      printf("couldnot initialize");
      return -1;
    }
  
  cvNamedWindow("frame");
  // cvNamedWindow("imgHsv"); 
  cvNamedWindow("imgRGB");
  cvNamedWindow("threshed");
  
  while(1)
    {
	IplImage* frame =0;
	frame = cvQueryFrame(capture);
	if(!frame)
	  { 
	    printf("error"); 
	    break;
	  }
	
	IplImage* imgHsv =cvCreateImage(cvGetSize(frame),8,3);
	
	IplImage* imgRGB = cvCreateImage(cvGetSize(frame),8,3);
	IplImage* sub = cvCreateImage(cvGetSize(frame),8,3);
	IplImage* imgSue = cvCreateImage(cvGetSize(frame),8,1);
	IplImage* imgValue = cvCreateImage(cvGetSize(frame),8,1);
	IplImage* imgHue = cvCreateImage(cvGetSize(frame),8,1);
      
	
	cvCvtColor(frame,imgHsv,CV_BGR2HSV);
	cvSetMouseCallback("frame", mouse_callback , (void*) frame);
	
	IplImage* colorThreshed = GetThresholdedImage(frame);    
	
	// cvSub(frame,imgRGB,sub,NULL);
        
	cvSetImageCOI(imgHsv,1);
	cvCopy(imgHsv,imgHue,NULL);
	
	cvSetImageCOI(imgHsv,2);
	cvCopy(imgHsv,imgSue,NULL);
	
	cvSetImageCOI(imgHsv,3);
	cvCopy(imgHsv,imgValue,NULL);
	
	cvSetImageCOI(imgHsv,0);
      
	
	
	cvAddS(imgHsv,cvScalar(40,0,0),imgHsv,colorThreshed);
	
	
	cvCvtColor(imgHsv,imgRGB,CV_HSV2BGR);
	//  cvShowImage("Sue",imgSue);
	//cvShowImage("Value",imgValue);
	// cvShowImage("imgHsv",imgHsv);
	// cvShowImage("imgHue",imgHue);
	cvShowImage("frame",frame);
	cvShowImage("imgRGB",imgRGB);
	cvShowImage("threshed",colorThreshed);
	
	cvReleaseImage(&imgHsv);
	cvReleaseImage(&imgSue);  
	cvReleaseImage(&imgRGB);
      
	char c = cvWaitKey(33);
	if(c == 27)
	  break;
    }
  cvReleaseCapture(&capture);
  return 0;
}




void mouse_callback( int event, int x, int y, int flags, void* param )
{
  
  IplImage* image = (IplImage*) param;

  switch( event ) {
    
  case CV_EVENT_LBUTTONDOWN: {
    
    // printf(" X- %d, Y - %d \n",x,y);
    CvScalar pix;
    pix=cvGet2D(param,x,y);
    double b,g,r,h,s,v,min,max,delta;
    b=pix.val[0];
    g=pix.val[1];
    r=pix.val[2];
    max=maximum(b,g,r);
    min=minimum(b,g,r);
    
    v=max;
    delta = max-min;
    
    if(max != 0)
      s=delta/max;
    else s=0;

    if(r==max) h=(g-b)/delta;
    if(g==max) h=2+ (b-r)/delta;
    if(b==max) h=4+ (r-g)/delta;       
    h=h*60;
    if(h<0)
      h+=360;
     h=h/2;
    
       
    
    printf("B: %f g:%f R:%f \n",b,g,r);
    // printf("max: %f min:%f\n",max,min);
    printf("H: %f S:%f V:%f \n",h,s,v);
    
 hsvPix = cvScalar(h,s,v);
 
  }
    break; 
    
    
  }
}



IplImage* GetThresholdedImage(IplImage* img)
{

  
  IplImage* imgHSV = cvCreateImage(cvGetSize(img),8,3);
  cvCvtColor(img,imgHSV,CV_BGR2HSV);
  float h,s,v,lh,ls,lv,hh,hs,hv;
  h=hsvPix.val[0];  
  s=hsvPix.val[1];
  v=hsvPix.val[2];
  lh=h-10;
  hh=h+10;
  
  
  IplImage* imgThreshed = cvCreateImage(cvGetSize(img),8,1);
  cvInRangeS(imgHSV,cvScalar(lh,100,100),cvScalar(hh,255,255),imgThreshed);
  cvReleaseImage(&imgHSV);
  return imgThreshed;
}

float maximum(int x,int y,int z)
{
  if(x>y )
    { 
      if(x>z)
	return x;
      else 
	return z;
      
    }
  else if(x>z)
    return y;
  else if(y>z)
    return y;
  else 
    return z;
  
  
}
float minimum(int x,int y,int z)
{
  if(x<y )
    { 
      if(x<z)
	return x;
      else 
	return z;
      
    }
  else if(x<z)
    return y;
  else if(y<z)
    return y;
  else 
    return z;
	

  }
  
