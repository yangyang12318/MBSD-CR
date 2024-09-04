#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
using namespace std;
using namespace cv;

extern "C" 
{
	

  cv::Mat* nparray_to_mat(float* data, int width, int height)
    {
        cv::Mat* mat = new cv::Mat(height, width, CV_32F, data);
        return mat;
    }

  
  void integralImgSqDiff(cv::Mat* src, cv::Mat* dst, int Ds, int t1, int t2, int m1, int n1)
{

  // 检查输入指针是否为空
  if (src == nullptr || dst == nullptr) {
    throw std::invalid_argument("Input and output Mat pointers must not be NULL");
  }
  //计算图像A与图像B的差值图C
  cv::Mat Dist2 = (*src)(cv::Range(Ds, src->rows - Ds), cv::Range(Ds, src->cols - Ds)) - (*src)(cv::Range(Ds + t1, src->rows - Ds + t1), cv::Range(Ds + t2, src->cols - Ds + t2));
  float *Dist2_data;
  for (int i = 0; i < m1; i++)
  {
    Dist2_data = Dist2.ptr<float>(i);
    for (int j = 0; j < n1; j++)
    {
      Dist2_data[j] *= Dist2_data[j];  //计算图像C的平方图D
    }
  }
  integral(Dist2, *dst, CV_32F); //计算图像D的积分图
}


void findClosestPixel(Mat* src , Mat* sar, Mat* nearestPixel, int ds, int Ds)
{
  cv::Mat src_tmp;
  src->convertTo(src_tmp, CV_32F);
  int m = src_tmp.rows;
  int n = src_tmp.cols;
  int boardSize = Ds + ds;//边缘扩充的大小应该是ds+Ds
  Mat src_board;
  copyMakeBorder(src_tmp, src_board, boardSize, boardSize, boardSize, boardSize, BORDER_REFLECT);
  
  Mat average(m, n, CV_32FC1, 0.0);
  Mat sweight(m, n, CV_32FC1, 0.0);


  //这里是添加SAR影像的地方
   cv::Mat sar_tmp;
  sar->convertTo(sar_tmp, CV_32F);
   Mat sar_board;
  copyMakeBorder(sar_tmp, sar_board, boardSize, boardSize, boardSize, boardSize, BORDER_REFLECT);
  
      
  
  Mat nearestDist = cv::Mat::ones(m, n, CV_32F)*FLT_MAX;
  float h2 = 80;
  int d2 = (2 * ds + 1)*(2 * ds + 1); //这个是求MSE的时候的元素的个数，

  int m1 = src_board.rows - 2 * Ds;   //这个是积分图的行数
  int n1 = src_board.cols - 2 * Ds;   //这个是积分图的列数。就是边缘扩充的图-2*搜索窗口大小
  Mat St(m1, n1, CV_32FC1, 0.0);

  for (int t1 = -Ds; t1 <= Ds; t1++){ //搜索窗口的行
    for (int t2 = -Ds; t2 <= Ds; t2++){//搜索窗口的列
    	if(t2==0&&t1==0)
	{
		continue;}

      integralImgSqDiff(&src_board, &St, Ds, t1, t2, m1, n1);//St是积分图
  // cout << " St:\n" << St << endl;
      for (int i = 0; i < m; i++){//原图的行
          float *sweight_p = sweight.ptr<float>(i);
          float *average_p = average.ptr<float>(i);
          float *v_p = sar_board.ptr<float>(i + Ds + t1 + ds);
 
        
          int i1 = i + ds+1 ;   //row 这是积分图上对应的行
          float *St_p1 = St.ptr<float>(i1 + ds);
          float *St_p2 = St.ptr<float>(i1 - ds-1 );

          for (int j = 0; j < n; j++){//原图的列

             int j1 = j + ds + 1;   //col
          float Dist2 = (St_p1[j1 + ds] + St_p2[j1 - ds - 1]) - (St_p1[j1 - ds - 1] + St_p2[j1 + ds]);
		
          Dist2 /= (-d2*h2);
          
          float w = exp(Dist2);
          
          sweight_p[j] += w;
          average_p[j] += w * v_p[j + Ds + t2 + ds];
         }
         }
      }
    }
  
       average = average / sweight;
  average.convertTo(*nearestPixel, CV_32F);
   
}


  void mat_to_nparray(cv::Mat* mat, float* data) 
{
        std::memcpy(data, mat->data, mat->total() * mat->elemSize());
    }

void release_mat(cv::Mat* mat) {
        delete mat;
}

}



