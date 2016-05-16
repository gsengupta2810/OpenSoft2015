#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "opencv2/video/tracking.hpp"

#include <deque>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

#define min_pixels 1000
#define max_pixels 1000

Mat rgb2hsv(Mat img)
{
	Mat img1=img.clone();
	Mat img2(img.rows,img.cols,CV_8UC3,Scalar(0,0,0));
	cvtColor(img1,img2,CV_BGR2HSV);
	return img2; 
}

void grouping(Mat img)
{
	Mat img1=img.clone();
	vector<Mat> channels;
	split(img1, channels);
	deque<deque<Point> > cluster(360);
	//cout<<"chutiyapa "<<cluster.size()<<endl;
	for(int i=0;i<img.rows;i++)
	{
		for (int j = 0; j < img.cols; ++j)
		{
				int pixel=channels[0].at<uchar>(i,j);
				cluster[pixel].push_back(Point(i,j));
		}
	}
	deque<deque<Point> > cluster_final(37);
	int m=0;
	fstream file;
	file.open("cluster.txt",fstream::out);
	for (int i = 10; i <360; i+=10)
	{
		for (int j = -10; j < 10; j++)
		{
			for (int k = 0; k < cluster[i+j].size(); ++k)
			{
				if(m<36)
				cluster_final[m].push_back(cluster[i+j][k]);
			}	
			m++;
		}
	}
	// for (int i = 0; i < 36; ++i)
	// {
	// 	for (int j = 0; j <cluster[i].size(); ++j)
	// 	{
	// 		file<<"("<<cluster_final[i][j].x<<","<<cluster_final[i][j].y<<")";
	// 	}
	// 	file<<endl;
	// }
	//  file.close();
	deque<Mat> show;
	int n=0;
	for(int i=0;i<36;i++)
	{
		cout<<cluster_final[i].size()<<" ";	
		if(cluster_final[i].size()>min_pixels)	
		{
			cout<<cluster_final[i].size()<<" "<<i<<endl;
			Mat temp(img.rows,img.cols,CV_8UC1,Scalar(255));
			for (int k = 0; k < cluster_final[i].size(); ++k)
			{
				if(cluster_final[i][k].x<img.rows && cluster_final[i][k].x<img.rows )
					{
						temp.at<uchar>(Point(cluster_final[i][k].y,cluster_final[i][k].x))=0;
					}
			}	
			show.push_back(temp);
		}
		cout<<endl;				
	}
	for (int i = 0; i < show.size(); ++i)
	{
	  char winName[20];
	  sprintf(winName, "image %d", i);
	  cv::imshow(winName,show[i]);
	}
	return;
}

Mat clusterimage(Mat img)
{
	Mat src = img.clone();
  Mat samples(src.rows * src.cols, 3, CV_32F);
  for( int y = 0; y < src.rows; y++ )
    for( int x = 0; x < src.cols; x++ )
      for( int z = 0; z < 3; z++)
        samples.at<float>(y + x*src.rows, z) = src.at<Vec3b>(y,x)[z];


  int clusterCount = 8;
  Mat labels;
  int attempts = 5;
  Mat centers;
  kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers );


  Mat new_image( src.size(), src.type() );
  for( int y = 0; y < src.rows; y++ )
    for( int x = 0; x < src.cols; x++ )
    { 
      int cluster_idx = labels.at<int>(y + x*src.rows,0);
      new_image.at<Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
      new_image.at<Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
      new_image.at<Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);
    }
  imshow( "clustered image", new_image );
  return new_image;
}
Mat enhance(Mat img)
{
	int alpha=1.2,beta=10;
	Mat img1=img.clone();
	Mat image=img.clone();
	GaussianBlur(img1, image, cv::Size(0, 0), 3);
	addWeighted(img1, 1.5, image, -0.5, 0, image);
	// Mat new_image(img.rows,img.cols,CV_8UC3,Scalar(0,0,0));
	// for( int y = 0; y < img.rows; y++ )
 //    	   { for( int x = 0; x < img.cols; x++ )
	// 	 { 
	// 	 	for( int c = 0; c < 3; c++ )
	// 	      {
	// 	        new_image.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( alpha*( image.at<Vec3b>(y,x)[c] ) + beta ); 
	// 		  }
	//     	  }
	//     } 
 //    imshow("enhanced2",new_image);
 //    return new_image;
	imshow("enhanced2",image);
    return image;

}

int main()
{
	Mat img;
	img=imread("test6.png",CV_LOAD_IMAGE_COLOR);
	Mat enhanced=enhance(img);
	Mat img2=clusterimage(enhanced);
	//Mat img2=clusterimage(img);
	//Mat img1=rgb2hsv(img2);
	//grouping(img1);
	imshow("Window",img);
	//imshow("hsv",img1);
	waitKey(0);
}