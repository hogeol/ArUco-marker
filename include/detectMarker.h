#include <iostream>
#include "opencv2/aruco.hpp"
#include "opencv2/opencv.hpp"

#ifndef detectMarkers
#define detectMarkers

class detectMarker
{
public:
	bool find_center_points(std::vector<cv::Point2f> OutputCenterPointArray);
	bool drawSquare(cv::InputOutputArray image, cv::InputArray rvec, cv::InputArray tvec, float length = 5,cv::Scalar color = cv::Scalar(255,0,0));
	detectMarker(cv::Mat originalImage, cv::Mat camMatrix, cv::Mat distCoeffs) :minitialImage(originalImage), mCammatrix(camMatrix), mDistCoeffs(distCoeffs) {};
	cv::Mat image2binary(cv::Mat originalImage);
	void find_contours(cv::Mat binaryImage, std::vector<std::vector<cv::Point2f>> markers);
	/*마커를 회전하여 구하는 코드는 외부에서 참조하였습니다.*/
	void rotateMarker();
	void make_bit_matrix();
	cv::Mat getByteListFromBits(const cv::Mat& bits);
	/*마커를 비트매트릭스로 변환하는 것 까지 외부참조입니다.*/
	void drawMarkerIDNcontour(cv::Mat inputImage);
	void findPose(cv::Mat inputImage);
	bool identify(const cv::Mat& onlyBits, int& idx, int& rotation);
	std::vector<std::vector<cv::Point2f>> final_detectedMarkers;
	std::vector<int> markerids;

private:
	cv::Mat minitialImage;
	cv::Mat minput_gray_image;
	cv::Mat mCammatrix;
	cv::Mat mDistCoeffs;
	std::vector<cv::Mat> mbitMatrixs;
	std::vector<std::vector<cv::Point2f>> mMarkers;
	std::vector<cv::Mat> mdetectedMarkersImage;
	std::vector<std::vector<cv::Point2f>> mdetectedMarkers;
	std::vector<std::vector<cv::Point2f>> mfinal_detectedMarkers;
};

#endif detectMarkers