#ifndef MYDFT_H
#define MYDFT_H

#include<iostream>
#include<opencv2/opencv.hpp>
#include<cmath>

using namespace cv;

void My_DFT(Mat input_image, Mat& output_image, Mat& transform_array);

#endif // MYDFT_H
