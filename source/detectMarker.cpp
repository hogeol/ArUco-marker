#include "detectMarker.h"
#include "opencv2/aruco.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/hal/hal.hpp"
#include <iostream>

using namespace cv;
using namespace std;

extern unsigned char DICT_6X6_1000_BYTES[][4][5];

bool detectMarker::find_center_points(std::vector<cv::Point2f> OutputCenterPointArray)
{
	Point2f centerPoint(0, 0);
	for (int i = 0; i < mfinal_detectedMarkers.size(); i++)
	{
		Point2f pt1 = Point2f(mfinal_detectedMarkers.at(i).at(i));
		Point2f pt2 = Point2f(mfinal_detectedMarkers.at(i).at(i + 1));
		Point2f pt3 = Point2f(mfinal_detectedMarkers.at(i).at(i + 2));
		Point2f pt4 = Point2f(mfinal_detectedMarkers.at(i).at(i + 3));
		Point2f pts[] = { mfinal_detectedMarkers.at(i).at(i),mfinal_detectedMarkers.at(i).at(i + 1),mfinal_detectedMarkers.at(i).at(i + 2),mfinal_detectedMarkers.at(i).at(i + 3) };
	}

	return true;
}

bool detectMarker::drawSquare(cv::InputOutputArray image, cv::InputArray rvec, cv::InputArray tvec, float length, Scalar color)
{
	float new_length = length * 1.05;
	float half_length = length / 2.0;
	vector<Point3f> squarePoints;
	vector<Point3f> squarePoints_air;
	squarePoints.push_back(Point3f(half_length, half_length, 0));
	squarePoints.push_back(Point3f(half_length, -half_length, 0));
	squarePoints.push_back(Point3f(-half_length, -half_length, 0));
	squarePoints.push_back(Point3f(-half_length, half_length, 0));
	squarePoints_air.push_back(Point3f(half_length + 0.3, half_length + 0.3, 0.5));
	squarePoints_air.push_back(Point3f(half_length + 0.3, -half_length + 0.3, 0.5));
	squarePoints_air.push_back(Point3f(-half_length + 0.3, -half_length + 0.3, 0.5));
	squarePoints_air.push_back(Point3f(-half_length + 0.3, half_length + 0.3, 0.5));
	vector<Point2f> imagePoints;
	vector<Point2f> imagePoints_airline;
	projectPoints(squarePoints_air, rvec, tvec, mCammatrix, mDistCoeffs, imagePoints);
	line(image, imagePoints[0], imagePoints[1], color, 5);
	line(image, imagePoints[1], imagePoints[2], color, 5);
	line(image, imagePoints[2], imagePoints[3], color, 5);
	line(image, imagePoints[3], imagePoints[0], color, 5);
	projectPoints(squarePoints, rvec, tvec, mCammatrix, mDistCoeffs, imagePoints);
	line(image, imagePoints[0], imagePoints[1], color, 10);
	line(image, imagePoints[1], imagePoints[2], color, 10);
	line(image, imagePoints[2], imagePoints[3], color, 10);
	line(image, imagePoints[3], imagePoints[0], color, 10);
	projectPoints(squarePoints_air, rvec, tvec, mCammatrix, mDistCoeffs, imagePoints_airline);
	line(image, imagePoints[0], imagePoints_airline[0], color, 10);
	line(image, imagePoints[1], imagePoints_airline[1], color, 10);
	line(image, imagePoints[2], imagePoints_airline[2], color, 10);
	line(image, imagePoints[3], imagePoints_airline[3], color, 10);
	return true;
}

cv::Mat detectMarker::image2binary(cv::Mat originalImage)
{
	Mat output_binary_image;
	cvtColor(originalImage, minput_gray_image, COLOR_BGR2GRAY);
	threshold(minput_gray_image, output_binary_image, 155, 255, THRESH_BINARY_INV | THRESH_OTSU);
	//adaptiveThreshold(input_gray_image, output_binary_image, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 91, 7);
	return output_binary_image;
}

void detectMarker::find_contours(cv::Mat binaryImage, vector<vector<Point2f>> markers)
{
	vector<vector<Point>> contours;
	findContours(binaryImage, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

	vector<Point2f> approx;
	for (size_t i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true) * 0.02, true);
		if (
			approx.size() == 4 &&
			fabs(contourArea(Mat(approx))) > 5000 &&
			fabs(contourArea(Mat(approx))) < 50000 &&
			isContourConvex(Mat(approx))
			)
		{
			//drawContours(input_image, contours, i, Scalar(0, 255, 0), 2, LINE_AA);
			vector<cv::Point2f> points;
			for (int j = 0; j < 4; j++)
			{
				points.push_back(cv::Point2f(approx[j].x, approx[j].y));
				//cout << points.at(j);
			}
			cv::Point v1 = points[1] - points[0];
			cv::Point v2 = points[2] - points[0];

			double o = (v1.x * v2.y) - (v1.y * v2.x);
			if (o < 0.0)
				swap(points[1], points[3]);

			mMarkers.push_back(points);
			markers.push_back(points);
		}
	}
}

void detectMarker::rotateMarker()
{
	vector<Point2f> square_points;

	int marker_image_side_length = 80;

	square_points.push_back(cv::Point2f(0, 0));
	square_points.push_back(cv::Point2f(marker_image_side_length - 1, 0));
	square_points.push_back(cv::Point2f(marker_image_side_length - 1, marker_image_side_length - 1));
	square_points.push_back(cv::Point2f(0, marker_image_side_length - 1));

	Mat marker_image;
	for (int i = 0; i < mMarkers.size(); i++)
	{
		vector<Point2f> m = mMarkers[i];

		Mat PerspectiveTransformMatrix = getPerspectiveTransform(m, square_points);

		warpPerspective(minput_gray_image, marker_image, PerspectiveTransformMatrix, Size(marker_image_side_length, marker_image_side_length));

		threshold(marker_image, marker_image, 125, 255, THRESH_BINARY | THRESH_OTSU);

		int cellSize = marker_image.rows / 8; //cellSize = 1 (marker 6 + border 2)
		int white_cell_count = 0;
		for (int y = 0; y < 8; y++)
		{
			int inc = 7; //Check only first and last coloumns	
			if (y == 0 || y == 7) inc = 1;

			for (int x = 0; x < 8; x += inc)
			{
				int cellX = x * cellSize;
				int cellY = y * cellSize;
				cv::Mat cell = marker_image(Rect(cellX, cellY, cellSize, cellSize));

				int total_cell_count = countNonZero(cell);

				if (total_cell_count > (cellSize * cellSize) / 2)
					white_cell_count++;
			}
		}
		if (white_cell_count == 0) {
			mdetectedMarkers.push_back(m);
			Mat img = marker_image.clone();
			mdetectedMarkersImage.push_back(img);
		}
	}
}

void detectMarker::make_bit_matrix()
{
	for (int i = 0; i < mdetectedMarkersImage.size(); i++)
	{
		Mat marker_image = mdetectedMarkersImage[i];
		Mat bitMatrix = Mat::zeros(6, 6, CV_8UC1);

		int cellSize = marker_image.rows / 8;
		for (int y = 0; y < 6; y++)
		{
			for (int x = 0; x < 6; x++)
			{
				int cellX = (x + 1) * cellSize;
				int cellY = (y + 1) * cellSize;
				Mat cell = marker_image(cv::Rect(cellX, cellY, cellSize, cellSize));

				int total_cell_count = countNonZero(cell);

				if (total_cell_count > (cellSize * cellSize) / 2)
					bitMatrix.at<uchar>(y, x) = 1;
			}
		}
		mbitMatrixs.push_back(bitMatrix);
	}
}

void detectMarker::drawMarkerIDNcontour(cv::Mat inputImage)
{
	vector<int> markerID;

	for (int i = 0; i < mdetectedMarkers.size(); i++)
	{
		Mat bitMatrix = mbitMatrixs[i];
		vector<Point2f> m = mdetectedMarkers[i];

		int rotation;
		int marker_id;
		if (!identify(bitMatrix, marker_id, rotation))
			cout << "Cannot find Corners" << endl;
		else {
			if (rotation != 0)
				std::rotate(m.begin(), m.begin() + 4 - rotation, m.end());

			cornerSubPix(minput_gray_image, m, Size(5, 5), Size(-1, -1), TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 30, 0.01));
			markerID.push_back(marker_id);
			markerids.push_back(marker_id);
			mfinal_detectedMarkers.push_back(m);
			final_detectedMarkers.push_back(m);
			aruco::drawDetectedMarkers(inputImage, mfinal_detectedMarkers, markerID, cv::Scalar(255, 0, 0));
		}
	}
	for (int i = 0; i < mfinal_detectedMarkers.size(); i++)
	{
			cout << i << ", "<< mfinal_detectedMarkers.at(i) << endl;
	}
}

void detectMarker::findPose(cv::Mat inputImage)
{
	vector<cv::Point3f> markerCorners3d;
	markerCorners3d.push_back(cv::Point3f(-0.5f, 0.5f, 0));
	markerCorners3d.push_back(cv::Point3f(0.5f, 0.5f, 0));
	markerCorners3d.push_back(cv::Point3f(0.5f, -0.5f, 0));
	markerCorners3d.push_back(cv::Point3f(-0.5f, -0.5f, 0));
	Mat rotation_vector, translation_vector;
	for (int i = 0; i < mfinal_detectedMarkers.size(); i++)
	{
		vector<Point2f> m = mfinal_detectedMarkers[i];
		solvePnP(markerCorners3d, m, mCammatrix, mDistCoeffs, rotation_vector, translation_vector);
		aruco::drawAxis(inputImage, mCammatrix, mDistCoeffs, rotation_vector, translation_vector, 1.0);
	
		//정육면체 그리는 코드입니다.
		drawSquare(inputImage, rotation_vector, translation_vector, 1, Scalar(255,0,0));
	}
		
}

bool detectMarker::identify(const cv::Mat& onlyBits, int& idx, int& rotation)
{
	int markerSize = 6;
	Mat dictionary = Mat(250, (6 * 6 + 7) / 8, CV_8UC4, (uchar*)DICT_6X6_1000_BYTES);
	//비트 매트릭스를 바이트 리스트로 변환합니다. 
	Mat candidateBytes = getByteListFromBits(onlyBits);

	idx = -1; // by default, not found

	//dictionary에서 가장 근접한 바이트 리스트를 찾습니다. 
	int MinDistance = markerSize * markerSize + 1;
	rotation = -1;
	for (int m = 0; m < dictionary.rows; m++) {

		//각 마커 ID
		for (unsigned int r = 0; r < 4; r++) {
			int currentHamming = hal::normHamming(
				dictionary.ptr(m) + r * candidateBytes.cols,
				candidateBytes.ptr(),
				candidateBytes.cols);

			//이전에 계산된 해밍 거리보다 작다면 
			if (currentHamming < MinDistance) {
				//현재 해밍 거리와 발견된 회전각도를 기록합니다. 
				MinDistance = currentHamming;
				rotation = r;
				idx = m;
			}
		}
	}

	//idx가 디폴트값 -1이 아니면 발견된 것
	return idx != -1;
}


cv::Mat detectMarker::getByteListFromBits(const cv::Mat& bits)
{
	int nbytes = (bits.cols * bits.rows + 8 - 1) / 8;

	Mat candidateByteList(1, nbytes, CV_8UC1, Scalar::all(0));
	unsigned char currentBit = 0;
	int currentByte = 0;

	uchar* rot0 = candidateByteList.ptr();

	for (int row = 0; row < bits.rows; row++)
	{
		for (int col = 0; col < bits.cols; col++)
		{
			rot0[currentByte] <<= 1;

			rot0[currentByte] |= bits.at<uchar>(row, col);

			currentBit++;
			if (currentBit == 8) {
				currentBit = 0;
				currentByte++;
			}
		}
	}
	return candidateByteList;
}

