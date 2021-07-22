#include <cv.h>
#include <highgui.h>
 
//Filter 2D example program
using namespace cv;

int main(int argc, char** argv) {
	CvCapture* capture = cvCaptureFromCAM(0);
 

		

       
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
			  
	
	
		Mat sobelFilter = Mat(3,3,CV_32FC1,sobelkernel);

 
	Mat filter = Mat(
			5,
			5,
			CV_32FC1,
			kernel);
 
	cvNamedWindow("filtered", CV_WINDOW_AUTOSIZE);
	//	cvQueryFrame(capture);	
 
	

	Mat frame;

 
	while(1) {
	  Mat dst,sobeldst,mul,tmp;

		frame = cvQueryFrame(capture);
		//	dst = cvCreateImage(cvGetSize(frame), 8, 1);
		//sobeldst= cvCreateImage(cvGetSize(frame),8,1);

	
		cvtColor(frame, dst, CV_BGR2GRAY);
		cvtColor(frame,sobeldst,CV_BGR2GRAY);		
		//cvCopy(dst,sobeldst);
		//convolution 
		// blur series 1
		medianBlur(dst,dst,1);


		imshow("blur1",dst);
		
		filter2D(dst,sobeldst,0,filter,Point(-1,-1),0,1);
		imshow("filtered", dst);
		
		mul = sobeldst;
		tmp = sobeldst;
		imshow("mul1",mul);
		
		// blur series 2	
		medianBlur(dst,dst,3);
		
		
		imshow("blur2",dst);
		
		filter2D(dst,sobeldst,0,filter,Point(-1,-1),0,1);
		imshow("filtered", dst);
		
		mul = tmp.cross(sobeldst);
		cvGEMM(mul,sobeldst,1,NULL,0,tmp,0);
		imshow("mul2",mul);
		
		





















	
		vector<Vec4i> lines;
		//	 HoughLinesP(dst,lines,/*resolution of param 'r' */1 ,/*resolution param theta*/CV_PI/180   ,/*minimum no. of intersections */100   ,/* min no. of points that can form a line*/90    ,/*max gap b/w 2 points to be considered in same line */30);
		/*		 for( size_t i=0; i<lines.size();i++)
     {
       Vec4i l = lines[i];
       line(frame , Point(l[0],l[1]) , Point(l[2],l[3]) , Scalar(0,0255) , 3, CV_AA);

       }*/

		 imshow("hough" , frame);

		

		//	cvFilter2D(sobeldst,sobeldst,&sobelFilter,cvPoint(-1,-1));
  //		filter2D(sobeldst,sobeldst,0,sobelFilter,Point(-1,-1),0,1);
  //		imshow("sobel",sobeldst);
		//	cvReleaseImage(&dst);
		
		//	vector<Vec4i> lines;
  //HoughLinesP(sobeldst,lines,/*resolution of param 'r' */1 ,/*resolution param theta*/CV_PI/180   ,/*minimum no. of intersections */70   ,/* min no. of points that can form a line*/50    ,/*max gap b/w 2 points to be considered in same line */30);
  /* for( size_t i=0; i<lines.size();i++)
     {
       Vec4i l = lines[i];
       line(sobeldst , Point(l[0],l[1]) , Point(l[2],l[3]) , Scalar(0,0255) , 3, CV_AA);

     }

     imshow("hough" , sobeldst);*/
  sobeldst.release();
  dst.release();
	
		
		int c = waitKey(10);
		if((char) c == 27)
		  break;
	}
	//	releaseCapture(&capture);
	cvReleaseCapture(&capture);

	
//cvDestroyWindow("filtered");
	
	return 0;
}
