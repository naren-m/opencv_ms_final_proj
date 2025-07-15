#include <cv.h>
#include "highgui.h"

int main(int argc, char** argv)
{
    //load image from directory
    IplImage* img = cvLoadImage("C:\\Users\\narenuday\\Desktop\\ram.jpg");


    IplImage* gray = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
    CvMemStorage* storage = cvCreateMemStorage(0);

    //covert to grayscale
    cvCvtColor(img, gray, CV_BGR2GRAY);

    // This is done so as to prevent a lot of false circles from being detected
    cvSmooth(gray, gray, CV_GAUSSIAN, 7, 7);

    IplImage* canny = cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
    IplImage* rgbcanny = cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,3);
    cvCanny(gray, canny, 50, 100, 3);

    //detect circles
    CvSeq* circles = cvHoughCircles(gray, storage, CV_HOUGH_GRADIENT, 1, 35.0, 75, 60,0,0);
    cvCvtColor(canny, rgbcanny, CV_GRAY2BGR);

    //draw all detected circles
    for (int i = 0; i < circles->total; i++)
    {
         // round the floats to an int
         float* p = (float*)cvGetSeqElem(circles, i);
         cv::Point center(cvRound(p[0]), cvRound(p[1]));
         int radius = cvRound(p[2]);
         //uchar* ptr;
         //ptr = cvPtr2D(img, center.y, center.x, NULL);
         //printf("B: %d G: %d R: %d\n", ptr[0],ptr[1],ptr[2]);
         CvScalar s;

         s = cvGet2D(img,center.y, center.x);//colour of circle
        printf("B: %f G: %f R: %f\n",s.val[0],s.val[1],s.val[2]);

         // draw the circle center
         cvCircle(img, center, 3, CV_RGB(0,255,0), -1, 8, 0 );

         // draw the circle outline
         cvCircle(img, center, radius+1, CV_RGB(0,0,255), 2, 8, 0 );

         //display coordinates
         printf("x: %d y: %d r: %d\n",center.x,center.y, radius);

    }

    //create window
    //cvNamedWindow("circles", 1);
    cvNamedWindow("SnookerImage", 1);
    //show image in window
    //cvShowImage("circles", rgbcanny);
    cvShowImage("SnookerImage", img);

    cvSaveImage("out.png", img);
    //cvDestroyWindow("SnookerImage");
    //cvDestroyWindow("circles");
    //cvReleaseMemStorage("storage");
    cvWaitKey(0);

    return 0;
}
