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
#include"PolyfitEigen.hpp"

#define SQR(x) ((x)*(x))

namespace fs = std::filesystem;

// Program configuration parameters.
std::string folder = "TestCases\\Case1_Ca_0.052";
bool speed_test = true;
// Experimental parameters. mu*U in experiment, which represents the viscous force of fluid.
float muU = 0.028;		//Experimental Parameter for case1
//float muU = 0.015;		//Experimental Parameter for case2

float tubewidth;
cv::Mat background;
cv::Mat kernel = (cv::Mat_<unsigned char>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);
cv::Mat kernel1 = (cv::Mat_<unsigned char>(3, 3) << 0, 1, 0, 1, 1, 1, 0, 1, 0);
cv::Mat empty_mat = (cv::Mat_<unsigned char>(1, 1) << 0);
cv::Mat inputframe, tempframe;
int num_frames;
const int npoints = 61;  //Number of boundary points for vector input

// Load all trained models
auto ca_model = fdeep::load_model("camodel.json");
auto c_model = fdeep::load_model("cmodel.json");
auto law_model = fdeep::load_model("lawmodel.json");
auto p_model = fdeep::load_model("pmodel.json");
// Global variables to store all predicted values
std::vector<float> all_ca;
std::vector<float> all_c;
std::vector<float> all_p;
std::vector<float> ks;
std::vector<float> gs;
std::vector<int> law;

template <class T>
std::vector<T> pw_interpolate(std::vector<T> xcoord, std::vector<T> ycoord, std::vector<T> data, int npoints)
{
	// Piece wise interpolation function. This function uses 7 points on the capsule boundary to get a quadratic fitted boundary.
	// Input : sorted coordinates of X and Y on the boundary, data on the boundary and number of interpolated points when output
	// Output: Vector of interpolated data.
	std::vector<T> interpolated;
	double totallength = 0;
	std::vector<double> lengthmark;
	lengthmark.push_back(0);
	for (int i = 0; i < xcoord.size() - 1; i++) {
		totallength += sqrt(SQR(xcoord[i] - xcoord[i + 1]) + SQR(ycoord[i] - ycoord[i + 1]));
		lengthmark.push_back(totallength);
	}
	lengthmark.push_back(lengthmark[lengthmark.size() - 1] + lengthmark[1] - lengthmark[0]);
	lengthmark.push_back(lengthmark[lengthmark.size() - 1] + lengthmark[2] - lengthmark[1]);
	lengthmark.push_back(lengthmark[lengthmark.size() - 1] + lengthmark[3] - lengthmark[2]);
	lengthmark.insert(lengthmark.begin(), lengthmark[0] - lengthmark[lengthmark.size() - 4] + lengthmark[lengthmark.size() - 5]);
	lengthmark.insert(lengthmark.begin(), lengthmark[0] - lengthmark[lengthmark.size() - 5] + lengthmark[lengthmark.size() - 6]);
	lengthmark.insert(lengthmark.begin(), lengthmark[0] - lengthmark[lengthmark.size() - 6] + lengthmark[lengthmark.size() - 7]);
	data.push_back(data[1]);
	data.push_back(data[2]);
	data.push_back(data[3]);
	data.insert(data.begin(), data[data.size() - 4]);
	data.insert(data.begin(), data[data.size() - 5]);
	data.insert(data.begin(), data[data.size() - 6]);
	std::vector<double> newdistance;
	for (int i = 0; i < npoints; i++) {
		newdistance.push_back((double)i / (double)(npoints - 1) * totallength);
	}
	int temp = -50;
	std::vector<T> tpoly;
	for (int i = 0; i < npoints; i++) {
		std::vector<double> dist;
		for (int j = 0; j < lengthmark.size(); j++) {
			dist.push_back(abs(newdistance[i] - lengthmark[j]));
		}
		std::vector<double>::iterator result = std::min_element(dist.begin(), dist.end());
		int index = std::distance(dist.begin(), result);
		if (index == temp) {
			interpolated.push_back(tpoly[0] + tpoly[1] * newdistance[i] + tpoly[2] * SQR(newdistance[i]));
		}
		else {
			std::vector<T> segmentdata(data.begin() + index - 3, data.begin() + index + 3);
			std::vector<T> segmentlm(lengthmark.begin() + index - 3, lengthmark.begin() + index + 3);
			std::vector <T> poly = polyfit_Eigen(segmentlm, segmentdata, 2);
			tpoly = poly;
			interpolated.push_back(poly[0] + poly[1] * newdistance[i] + poly[2] * SQR(newdistance[i]));
		}
		temp = index;
	}
	return interpolated;
}

void main_process(cv::Mat img)
{
	//Main process function of the Real-time Characterisation of microcapsules.
	//Input : Image in OpenCV Matrix
	cv::Mat frame;
	cv::Mat threshimg, im_floodfill, im_floodfill_inv;
	cv::Mat contourpts, polar_r, polar_theta, idx, sorted_bound;
	double area, perimeter;

	//Obtain the image and substract with background. If not bright enough, then the image is empty.
	img.copyTo(frame);
	frame = abs(frame - background);
	cv::threshold(frame, threshimg, 30, 255, cv::THRESH_BINARY);
	cv::Scalar totbright = cv::sum(threshimg);
	if (totbright(0) < 10000)
	{
		std::cout << "frame " << num_frames << " no particle detected" << std::endl;
		num_frames++;
		return;
	}
	// Denoise the image and extract boundary points
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
	// Process all capsules on this image
	for (int ci = 0; ci < contours.size(); ci++)
	{
		boundbox = cv::boundingRect(contours[ci]);
		area = cv::contourArea(contours[ci]);
		perimeter = cv::arcLength(contours[ci], true);
		if (perimeter < 40)
			continue;
		num_particle++;
		// If aspect ratio is too high, then not valid
		if (((float)boundbox.width / (float)boundbox.height > 2) || ((float)boundbox.width / (float)boundbox.height < 0.5))
		{
			std::cout << "Particle " << num_particle << " at frame " << num_frames << " is over skew " << std::endl;
			continue;
		}
		// Centred, sorted and interpolated boundary nodes.
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
		std::vector<double> x_coor_int, y_coor_int;
		x_coor_int = pw_interpolate(x_coor, y_coor, x_coor, npoints);
		y_coor_int = pw_interpolate(x_coor, y_coor, y_coor, npoints);

		cx = 0;
		cy = 0;
		for (int cp = 0; cp < x_coor_int.size(); cp++) {
			cx += x_coor_int[cp];
			cy += y_coor_int[cp];
		}
		cx = cx / x_coor_int.size();
		cy = cy / y_coor_int.size();

		for (int cp = 0; cp < x_coor_int.size(); cp++) {
			x_coor_int[cp] = x_coor_int[cp] - cx;
			y_coor_int[cp] = y_coor_int[cp] - cy;
		}

		//Create the input vector for MLP
		float input_vec[2 * npoints];
		fdeep::tensor input_tensor(fdeep::tensor_shape(npoints * 2), 0.0f);
		for (int cp = 0; cp < npoints; cp++)
		{
			input_tensor.set(fdeep::tensor_pos(cp), x_coor_int[cp] / tubewidth);
			input_tensor.set(fdeep::tensor_pos(cp + npoints), y_coor_int[cp] / tubewidth);
		}

		//MLP prediction and store values in global variables
		auto law_result = law_model.predict({ input_tensor });
		auto law_values = law_result[0].to_vector();
		auto ca_result = ca_model.predict({ input_tensor });
		auto ca_values = ca_result[0].to_vector();
		ca_values[0] = ca_values[0];
		float capsule_ks = muU / ca_values[0];
		if (law_values[2] > law_values[1] && law_values[2] > law_values[0]) {
			auto p_result = p_model.predict({ input_tensor });
			auto p_values = p_result[0].to_vector();
			float capsule_gs = capsule_ks * (1 - p_values[0]) / (1 + p_values[0]);
			ks.push_back(capsule_ks);
			gs.push_back(capsule_gs);
			all_ca.push_back(ca_values[0]);
			all_p.push_back(p_values[0]);
			all_c.push_back(0);
			std::cout << "Constitutive law is Hooke's law\t" << ca_values[0]<<"\npredict Ks:\t" << capsule_ks << "\npredict Gs:\t" << capsule_gs << std::endl;
		}
		else if(law_values[1] > law_values[2] && law_values[1] > law_values[0]) {
			auto c_result = c_model.predict({ input_tensor });
			auto c_values = c_result[0].to_vector();
			float capsule_gs = capsule_ks / (1 + 2*c_values[0]);
			ks.push_back(capsule_ks);
			gs.push_back(capsule_gs);
			all_ca.push_back(ca_values[0]);
			all_p.push_back(0);
			all_c.push_back(c_values[0]);
			std::cout << "Constitutive law is SK law\t" << ca_values[0] << "\npredict Ks:\t" << capsule_ks << "\npredict Gs:\t" << capsule_gs << std::endl;
		}
		else if (law_values[0] > law_values[2] && law_values[0] > law_values[1]) {
			all_ca.push_back(ca_values[0]);
			float capsule_gs = capsule_ks / 3;
			ks.push_back(capsule_ks);
			gs.push_back(capsule_gs);
			all_p.push_back(0);
			all_c.push_back(0);
			std::cout << "Constitutive law is NH law\t" << ca_values[0] << "\npredict Ks:\t" << capsule_ks << "\npredict Gs:\t" << capsule_gs << std::endl;
		}

		
	}
	num_frames++;
	return;
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
	background = cv::imread("TestCases\\Case1_Ca_0.052\\00000.png");
	cv::cvtColor(background, background, cv::COLOR_BGR2GRAY);
	cv::Scalar avgbright = cv::mean(background);
	std::cout << background.size << std::endl;
	std::cout << avgbright << std::endl;
	int threshold = avgbright(0) * 0.78;
	background.copyTo(temp_bak);
	cv::threshold(temp_bak, temp_bak, threshold, 255, cv::THRESH_BINARY_INV);
	cv::Mat tubecol = temp_bak.col(10);
	int testpoint = tubecol.at<unsigned char>(10);

	cv::Size colsize = tubecol.size();
	int temp1, temp2, temp3, upperline = 0, lowerline = 0;
	bool flag = false;

	//find tube edge and determine tube diameter (in pixels)
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
				upperline = (temp1 + temp2) / 2 -1;
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
				lowerline = (temp1 + temp2) / 2 + 1;
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
	
	// Load all data into memory
	int counter = 0;
	for (const auto& entry : fs::directory_iterator(folder))
	{
		img = cv::imread(entry.path().string());
		cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
		all_img.push_back(img);
	}
	const auto start_time1 = std::chrono::high_resolution_clock::now();
	std::vector<std::thread> threads;

	// Speed benchmark, will process the image 10000 times. Only for speed performance benchmark
	if (speed_test) {
		for (int i = 0; i < 10000; i++)
		{
			main_process(all_img[1]);
		}
		std::cout << "Total Test time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time1).count() << " ms " << std::endl;
		std::cout << "Average Time per frame: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time1).count() / 10000 << " ms " << std::endl;
		std::cout << "Finished!" << std::endl;
	}

	// Run normally, process all images within the folder
	else {
		for (int i = 0; i < all_img.size(); i++)
		{
			main_process(all_img[i]);
		}
	}
	return 0;
}
