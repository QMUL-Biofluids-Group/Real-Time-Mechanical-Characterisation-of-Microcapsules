#pragma once
#include <future>
#include <deque>
#include <iostream>
#include <chrono>
#include <string>
#include<opencv2/opencv.hpp>
#include<filesystem>
#include<vector>
#include<thread>
#include"thread_pool.hpp"
#include<fdeep/fdeep.hpp>
#include<spline.h>

namespace fs = std::filesystem;

float temp_arr[301][301];
float tubewidth;
cv::Mat background;
cv::Mat kernel = (cv::Mat_<unsigned char>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);
cv::Mat kernel1 = (cv::Mat_<unsigned char>(3, 3) << 0, 1, 0, 1, 1, 1, 0, 1, 0);
cv::Mat empty_mat = (cv::Mat_<unsigned char>(1, 1) << 0);
cv::Mat inputframe, tempframe;
int num_frames;
const int npoints = 61;  //boundary points for vector input
auto ca_model = fdeep::load_model("ca_test.json");
auto c_model = fdeep::load_model("c_test.json");
auto size_model = fdeep::load_model("size_test.json");
fdeep::tensor inputtensor = fdeep::tensor(fdeep::tensor_shape(301, 301, 1), 0);

void process_test(cv::Mat img)
{
	cv::Mat frame;
	cv::Mat threshimg, im_floodfill, im_floodfill_inv;
	cv::Mat contourpts, polar_r, polar_theta, idx, sorted_bound;
	double area, perimeter;
	img.copyTo(frame);
	frame = abs(frame - background);
	cv::threshold(frame, threshimg, 12, 255, cv::THRESH_BINARY);
	cv::Scalar totbright = cv::sum(threshimg);
	if (totbright(0) < 10000)
	{
		std::cout << "frame " << num_frames << " no particle detected" << std::endl;
		num_frames++;
		return;
	}
	cv::erode(threshimg, threshimg, kernel);
	cv::dilate(threshimg, threshimg, kernel);
	cv::dilate(threshimg, threshimg, kernel);
	im_floodfill = threshimg.clone();
	cv::floodFill(im_floodfill, cv::Point(0, 0), cv::Scalar(255));
	cv::bitwise_not(im_floodfill, im_floodfill_inv);
	frame = (threshimg | im_floodfill_inv);
	cv::erode(frame, frame, kernel);
	std::vector<std::vector<cv::Point>>contours;
	std::vector<cv::Vec4i> hierarchy;
	std::vector<std::vector<cv::Point>> contour1;
	std::vector<cv::Vec4i> hierarchy1;
	cv::findContours(frame, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
	if (contours.empty())
	{
		num_frames++;
		return;
	}
	cv::Rect boundbox;
	cv::Mat roi;

	int num_particle = 0;
	for (int ci = 0; ci < contours.size(); ci++)
	{
		boundbox = cv::boundingRect(contours[ci]);
		std::cout << boundbox << std::endl;
		area = cv::contourArea(contours[ci]);
		perimeter = cv::arcLength(contours[ci], true);
		if (perimeter < 40)
			continue;
		num_particle++;
		std::cout << "Particle " << num_particle << " at frame " << num_frames << " has area " << area << " and perlimeter " << perimeter << std::endl;
		roi = frame(boundbox);
		if ((boundbox.width / boundbox.height > 2) || (boundbox.width / boundbox.height < 0.5))
		{
			std::cout << "Particle " << num_particle << " at frame " << num_frames << " is over skew " << std::endl;
			continue;
		}


		contourpts = cv::Mat(contours[ci]).reshape(1);
		contourpts.convertTo(contourpts, CV_32F);
		cv::Moments M = cv::moments(contours[ci]);
		float cx = M.m10 / M.m00;
		float cy = M.m01 / M.m00;
		contourpts.col(0) = contourpts.col(0) - cx;
		contourpts.col(1) = contourpts.col(1) - cy;
		cv::cartToPolar(contourpts.col(0), contourpts.col(1), polar_r, polar_theta);			
		cv::sortIdx(polar_theta, idx, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
		sorted_bound = contourpts;
		std::vector<double> x_coor, y_coor, temp;
		for (int cp = 0; cp < idx.rows; cp++)
		{
			x_coor.push_back(contourpts.at<float>(idx.at<int>(cp), 0));
			y_coor.push_back(contourpts.at<float>(idx.at<int>(cp), 1));
			temp.push_back(cp);
		}
		std::cout << "tpx" << x_coor[0] << std::endl;
		std::cout << "tpy" << y_coor[0] << std::endl;
		tk::spline sx(temp, x_coor, tk::spline::cspline_hermite);
		tk::spline sy(temp, y_coor, tk::spline::cspline_hermite);
		float input_vec[2 * npoints];
		fdeep::tensor input_tensor(fdeep::tensor_shape(npoints * 2, 1), 0.0f);
		for (int cp = 0; cp < npoints; cp++)
		{
			input_vec[cp] = (sx(cp) - cx) / tubewidth;
			input_tensor.set(fdeep::tensor_pos(cp, 0), (sx(cp)) / tubewidth);
			input_vec[cp + npoints] = (sy(cp) - cy) / tubewidth;
			input_tensor.set(fdeep::tensor_pos(cp + npoints, 0), (sy(cp)) / tubewidth);
		}
		auto size_result = size_model.predict({ input_tensor });
		auto ca_result = ca_model.predict({ input_tensor });
		auto c_result = c_model.predict({ input_tensor });
		std::cout << "predict_size\t" << fdeep::show_tensors(size_result) << "\npredict_ca\t" << fdeep::show_tensors(ca_result) << "\npredict_C\t" << fdeep::show_tensors(c_result) << std::endl;
	}
	num_frames++;
	return;
}


template<typename T>
void pop_front(std::vector<T>& vec)
{
	assert(!vec.empty());
	vec.front() = std::move(vec.back());
	vec.pop_back();
}

// The entry point to the app.
int main()
{
	std::vector<cv::Mat> all_img;
	cv::Mat img;
	cv::Mat frame;
	cv::Mat temp_bak;
	cv::Mat threshimg, im_floodfill, im_floodfill_inv;
	cv::Mat roi, roierode;
	cv::Mat contourpts, polar_r, polar_theta;
	cv::Mat idx;
	cv::Mat sorted_bound;
	cv::Mat display_frame;
	double area, perimeter;
	img.copyTo(frame);
	background = cv::imread("DataSample\\00000.png");
	cv::cvtColor(background, background, cv::COLOR_BGR2GRAY);
	cv::Scalar avgbright = cv::mean(background);
	std::cout << background.size << std::endl;
	std::cout << avgbright << std::endl;
	int threshold = avgbright(0) * 0.78;
	background.copyTo(temp_bak);
	cv::threshold(temp_bak, temp_bak, threshold, 255, cv::THRESH_BINARY_INV);
	cv::Mat tubecol = temp_bak.col(10);
	std::cout << "tp0" << std::endl;
	int testpoint = tubecol.at<unsigned char>(10);

	cv::Size colsize = tubecol.size();
	int temp1, temp2, temp3, upperline = 0, lowerline = 0;
	bool flag = false;

	//find tube edge
	for (int i = 0; i < colsize.height - 1; i++)
	{
		testpoint = tubecol.at<unsigned char>(i + 1) - tubecol.at<unsigned char>(i);
		if (testpoint > 10)
		{
			temp1 = i + 1;
			flag = true;
		}
		if (testpoint < -10)
		{
			temp2 = i;
			if (flag)
			{
				upperline = (temp1 + temp2) / 2;
				break;
			}
			else
			{
				std::cout << "upperline error please redefine camera ROI, tube value is " << tubecol << std::endl;
				exit;
			}
		}
	}

	flag = false;
	tubecol.at<unsigned char>(colsize.height - 1) = 0;
	for (int i = colsize.height - 1; i > 0; i--)
	{

		testpoint = tubecol.at<unsigned char>(i - 1) - tubecol.at<unsigned char>(i);
		if (testpoint > 10)
		{
			temp1 = i - 1;
			flag = true;
		}
		if (testpoint < -10)
		{
			temp2 = i;
			if (flag)
			{
				lowerline = (temp1 + temp2) / 2;
				break;
			}
			else
			{
				std::cout << "lowerline error please redefine camera ROI, tube value is " << tubecol << std::endl;
				exit;
			}
		}
	}

	tubewidth = lowerline - upperline;
	inputframe = cv::Mat::ones(tubewidth, tubewidth, CV_8UC1);
	tempframe = cv::Mat::ones(tubewidth, tubewidth, CV_8UC1);
	std::cout << "up" << upperline << "dn" << lowerline << std::endl;
	std::cout << "tubewidth is " << tubewidth << std::endl;
	std::string folder = "test10";
	img = cv::imread("1raw.png");
	//cv::imshow("test", img);
	cv::waitKey(1);
	int counter = 0;
	for (const auto& entry : fs::directory_iterator(folder))
	{
		/*	if (counter > 500)
				break;
			counter++;*/
		img = cv::imread(entry.path().string());
		cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
		all_img.push_back(img);
	}
	const auto start_time1 = std::chrono::high_resolution_clock::now();
	std::vector<std::thread> threads;

	int num_threads = 1;
	thread_pool pool(num_threads);

	//	for (int k = 0; k < 10; k++)
	for (int i = 0; i < all_img.size(); i++)
	{
		pool.push_task(process_test, all_img[i]);
	}
	pool.wait_for_tasks();
	std::cout << "MT test time " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time1).count() << std::endl;
	std::cout << "Finished!" << std::endl;
	return 0;
}
