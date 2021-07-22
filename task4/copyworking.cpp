#include "cv.h"
#include "highgui.h"

using namespace cv;





#include <iostream>
#include <math.h>
#include <string.h>

using namespace std;



void help()
{
        cout <<
        "\nA program using pyramid scaling, Canny, contours, contour simpification and\n"
        "memory storage (it's got it all folks) to find\n"
        "squares in a list of images pic1-6.png\n"
        "Returns sequence of squares detected on the image.\n"
        "the sequence is stored in the specified memory storage\n"
        "Call:\n"
        "./squares\n"
    "Using OpenCV version %s\n" << CV_VERSION << "\n" << endl;
}


int thresh = 50, N = 2; // karlphillip: decreased N to 2, was 11.
const char* wndname = "Square Detection Demo";

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
void findSquares( const Mat& image, vector<vector<Point> >& squares )
{
   
}


// the function draws all the squares in the image
void drawSquares( Mat& image, const vector<vector<Point> >& squares )
{
   
}



int main()
{
  CvCapture* capture = cvCaptureFromCAM(0);
  Mat frame;
  while(1)
    {
      frame =cvQueryFrame(capture);
      imshow("show" , frame);

      // help();
 //  namedWindow( wndname, 1 );
     vector<vector<Point> > squares;

  
     //    Mat image = frame ;//imread(names[i], 1);
       

     //// find squares begin ////////////////////////////////////////////////////////////////////////////////////////////

 squares.clear();

    Mat pyr, timg, gray0(frame.size(), CV_8U), gray;

    // karlphillip: dilate the image so this technique can detect the white square,
    Mat out(image);
    dilate(out, out, Mat(), Point(-1,-1));
    // then blur it so that the ocean/sea become one big segment to avoid detecting them as 2 big squares.
    medianBlur(out, out, 7);


    // down-scale and upscale the image to filter out the noise
    pyrDown(out, pyr, Size(out.cols/2, out.rows/2));
    pyrUp(pyr, timg, out.size());
    vector<vector<Point> > contours;

    // find squares in every color plane of the image
    for( int c = 0; c < 3; c++ )
    {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        for( int l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if( l == 0 )
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 0, thresh, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;
            }

            // find contours and store them all as a list
            findContours(gray, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

            vector<Point> approx;

            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if( approx.size() == 4 &&
                    fabs(contourArea(Mat(approx))) > 1000 &&
                    isContourConvex(Mat(approx)) )
                {
                    double maxCosine = 0;

                    for( int j = 2; j < 5; j++ )
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if( maxCosine < 0.3 )
                        squares.push_back(approx);
                }
            }
        }
    }

     // / find square end ////////////////////////////////////////////////////////////////////////////////////////////////



    //findSquares(, squares);
      for( size_t i = 0; i < squares.size(); i++ )
    {
        const Point* p = &squares[i][0];
        int n = (int)squares[i].size();
        polylines(frame, &p, &n, 1, true, Scalar(0,255,0), 3, CV_AA);
    }

    imshow("square", frame); 

 drawSquares(image, squares);
      

      char c = waitKey(20);
      if(c == 27)
	break;
    }
  return 0;
}
