#include "cv.h"
#include "highgui.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
  Mat input = imread("/home/narenuday/Desktop/ram.jpg");

  Mat output,src_gray;
  Mat grad_x,grad_y;
  Mat abs_grad_x,abs_grad_y;
  Mat grad;
  
  int kernel_size = 3;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;
 
  //  GaussianBlur( input, input, Size(3,3), 0, 0, BORDER_DEFAULT );
  medianBlur(input,input,7);
  /// Convert it to gray
   cvtColor( input, src_gray, CV_RGB2GRAY );
  
    Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
   convertScaleAbs( grad_x, abs_grad_x );

  
  Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_y, abs_grad_y );

  /// Total Gradient (approximate)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

    imshow( "grad", grad );

  //imshow("output",output);
  imshow("input",input);
  waitKey();
  
  return 0;


}
