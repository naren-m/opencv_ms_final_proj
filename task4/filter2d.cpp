#include <cv.h>
#include <highgui.h>
 
//Filter 2D example program

int main(int argc, char** argv) {
	CvCapture* capture = cvCaptureFromCAM(0);
 

		
	/*	//works good for minute boarders 
		float kernel[] = {-4, 1, 4,
			   3, 0,-3,
			   4,-1,-4};

		
	// sharpening the image as the ancohor point wiil lwave with one when the matrix is summed up
	/*	float kernel[] = {0,-1,0,
			  -1,5,-1,
			  0,-1,0};*/

	/*	// works good on good edges as expected
	float kernel[] = { 0,1,2,
			   -1,0,1,
			   -2,-1,0};*/

	/*			// works on thick boarders only
	float kernel[] = { 1,2,1,
			   -2, -4, 2,
			   1,-2, 1}; */
 
	/*	// image become black completely
	float kernel[] = {0, 0, 0,
			   0, 0,0,
			   0,0, 0};*/

	// the middle row has something to deal with horizontal edges
	//This is not detecting horizontal edges as 
	/*	float kernel[] = { 4,0, -4,
			   0, 0,0,
			   -4,0, 4};*/

	// the middle column is for detecting horizontal edges
	/*	float kernel[] = { 0,2, 0,
			   0, 0,0,
			   0,-2, 0};*/

	// the middle row has something to deal with verticaledges
	/*	float kernel[] = { 0,0,0,
			   2,0,-2,
			   0,0,0};*/

	// both the edges we not clearly detected when corners are 1


	// vertical edges are clearly getting detected
	// bias on vertical

	// Result is not same as having -1 s in firs column and last column
	/*	float kernel[] = { 1,0, -1,
	                           0, 0, 0,
			           1,0, -1};  */
		//horizontal lines are cleary getting detected
	/*	float kernel[] = { 1, 0, -1,
	                    0, 0, 0,
			    -1, 0, 1};

	*/
	// Horizontal sobel filter
		float sobelkernel[] = { -1, -2, -1,
	                    0, 0, 0,
			    1, 2, 1};
	
	
		float kernel[] = {-2/85.0 ,-16/85.0 ,-32/85.0 ,-16/85.0 ,-2/85.0 ,
				  -8/85.0,-64/85.0 ,-128/85.0 ,-64/85.0 ,-8/85.0 ,
				  0 ,0 ,0 ,0 ,0 ,
				  8/85.0 ,64/85.0 ,128/85.0 ,64/85.0 ,8/85.0 ,
				  2/85.0 ,16/85.0 ,32/85.0 ,16/85.0 ,2/85.0 ,
			  };
			  
	
	
		CvMat sobelFilter = cvMat(3,3,CV_32FC1,sobelkernel);

 
	CvMat filter = cvMat(
			5,
			5,
			CV_32FC1,
			kernel);
 
	cvNamedWindow("filtered", CV_WINDOW_AUTOSIZE);
	cvQueryFrame(capture);	
 
	IplImage* frame = NULL;// cvCreateImage( cvSize( 640,420),IPL_DEPTH_8S,3   ); 

	IplImage* dst = 0;//cvCreateImage( cvGetSize(frame),IPL_DEPTH_8S,1);

	IplImage* sobeldst = 0;

 
	while(1) {
		frame = cvQueryFrame(capture);
		dst = cvCreateImage(cvGetSize(frame), 8, 1);
		sobeldst= cvCreateImage(cvGetSize(frame),8,1);
		cvCvtColor(frame, dst, CV_BGR2GRAY);
		cvCvtColor(frame,sobeldst,CV_BGR2GRAY);		
		//cvCopy(dst,sobeldst);
		//convolution 
		cvShowImage("gray",dst);
	
		cvFilter2D(dst, dst, &filter, cvPoint(-1,-1) );
		
		cvShowImage("filtered", dst);
		cvFilter2D(sobeldst,sobeldst,&sobelFilter,cvPoint(-1,-1));
		cvShowImage("sobel",sobeldst);
		cvReleaseImage(&dst);
			
		
		int c = cvWaitKey(10);
		if((char) c == 27)
		  break;
	}
	cvReleaseCapture(&capture);
	cvDestroyWindow("filtered");
	
	return 0;
}
