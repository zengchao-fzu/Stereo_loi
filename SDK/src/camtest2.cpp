#include <cv.h>
#include <highgui.h>
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include "loitorusbcam.h"
#include "loitorimu.h"

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
using namespace std;

bool close_img_viewer = false;
bool visensor_Close_IMU_viewer = false;

// 当前左右图像的时间戳
timeval left_stamp, right_stamp;

static int rate = 10;

static void saveXYZ(const char* filename, const cv::Mat& mat)
{
    const double max_z = 1.0e4;
    FILE* fp = fopen(filename, "wt");
    for(int y = 0; y < mat.rows; y++)
    {
        for(int x = 0; x < mat.cols; x++)
        {
            cv::Vec3f point = mat.at<cv::Vec3f>(y, x);
            if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
            fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
        }
    }
    fclose(fp);
}

/**
	* date:2017.5.29
	* valen
	* calculate the Disparity && display the depth
	* reference:
	* http://blog.csdn.net/chentravelling/article/details/70254682;
	* http://www.cnblogs.com/grandyang/p/5805261.html
	* http://docs.opencv.org/master/d9/dba/classcv_1_1StereoBM.html
*/
void *opencv_showimg(void*)
{
	cv::Mat img_left;
	cv::Mat img_right;

	if (!visensor_resolution_status)
	{
		img_left.create(cv::Size(640, 480), CV_8U);
		img_right.create(cv::Size(640, 480), CV_8U);
		img_left.data = new unsigned char[IMG_WIDTH_VGA * IMG_HEIGHT_VGA];
		img_right.data = new unsigned char[IMG_WIDTH_VGA * IMG_HEIGHT_VGA];
	}
	else
	{
		img_left.create(cv::Size(752, 480), CV_8U);
		img_right.create(cv::Size(752, 480), CV_8U);
		img_left.data = new unsigned char[IMG_WIDTH_WVGA * IMG_HEIGHT_WVGA];
		img_right.data = new unsigned char[IMG_WIDTH_WVGA * IMG_HEIGHT_WVGA];
	}
	while (!close_img_viewer)
	{
		if (visensor_cam_selection == 2)
		{
			visensor_imudata paired_imu = visensor_get_leftImg((char *)img_left.data, left_stamp);

			// 显示同步数据的时间戳（单位微秒）
			cout << "left_time : " << left_stamp.tv_usec << endl;
			cout << "paired_imu time ===== " << paired_imu.system_time.tv_usec << endl << endl;

			cv::imshow("left", img_left);
			cvWaitKey(1);
		}
		//Cam2
		else if (visensor_cam_selection == 1)
		{
			visensor_imudata paired_imu = visensor_get_rightImg((char *)img_right.data, right_stamp);

			// 显示同步数据的时间戳（单位微秒）
			cout << "right_time : " << right_stamp.tv_usec << endl;
			cout << "paired_imu time ===== " << paired_imu.system_time.tv_usec << endl << endl;

			cv::imshow("right", img_right);
			cvWaitKey(1);
		}
		// Cam1 && Cam2
		else if (visensor_cam_selection == 0)
		{
			visensor_imudata paired_imu = visensor_get_stereoImg((char *)img_left.data, (char *)img_right.data, left_stamp, right_stamp);

			// 显示同步数据的时间戳（单位微秒）
			// cout << "left_time : " << left_stamp.tv_usec << endl;
			// cout << "right_time : " << right_stamp.tv_usec << endl;
			// cout << "paired_imu time ===== " << paired_imu.system_time.tv_usec << endl << endl;

			// cv::imshow("left", img_left);
			// cv::imshow("right", img_right);

			//相机标定阶段
			cv::Mat left_m, left_d, right_m, right_d;
			left_m = (cv::Mat_<double>(3, 3) << 461.599631 , 0.000000, 385.421541,	0.000000, 461.949571 , 274.545319, 0.000000, 0.000000 , 1.000000);
			left_d = (cv::Mat_<double>(5, 1) << -0.384259, 0.122324, -0.000004, -0.001681, 0.000000);
			right_m = (cv::Mat_<double>(3, 3) << 465.330915, 0.000000, 380.598352, 0.000000, 466.862222, 204.123244, 0.000000 , 0.000000 , 1.000000);
			right_d = (cv::Mat_<double>(5, 1) << -0.369965, 0.102771, 0.002052, 0.003675, 0.000000);
			//立体校准优化
			cv::Mat disp;
			cv::Mat Q, R1, R2, P1, P2;
			cv::Rect roi_left, roi_right;
			cv::Mat T ;
			cv::Mat R ;
			T = (cv::Mat_<double>(3, 1) << -0.09830097401533426, 0.00014149632675107717, 0.007566402778860449);
			R = (cv::Mat_<double>(3, 3) << 0.9987819593989379, -0.0031174759824076997, 0.04924305964009116, 0.002459363552277384, 0.9999069297666924, 0.013419513235013242, -0.04928031158724743, -0.013282061136965045, 0.9986966695357593);
			cv::stereoRectify( left_m, left_d, right_m, right_d, img_left.size(), R, T, R1, R2, P1, P2, Q, 1, -1, img_left.size(), &roi_left, &roi_right);
			
			//立体匹配阶段
			cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(16, 9);
			//预处理起类型，降低亮度失真，消除噪声和增强纹理归一化的类型
			bm->setPreFilterType(CV_STEREO_BM_NORMALIZED_RESPONSE);
			//预处理滤波器窗口大小5*5 - 21*21之间
			bm->setPreFilterSize(9);
			//预处理滤波器的截断值，预处理的输出值仅保留[-preFilterCap, preFilterCap]范围内的值，参数范围：1 - 31
			bm->setPreFilterCap(31);
			//SAD窗口大小，容许范围是[5,255]，一般应该在 5x5 至 21x21 之间，参数必须是奇数
			bm->setBlockSize(9);
			//最小视差，默认值为 0, 可以是负值，int 型
			bm->setMinDisparity(0);
			//视差窗口，即最大视差值与最小视差值之差, 窗口大小必须是 16 的整数倍
			bm->setNumDisparities(64);
			//低纹理区域的判断阈值。如果当前SAD窗口内所有邻居像素点的x导数绝对值之和小于指定阈值，则该窗口对应的像素点的视差值为 0
			bm->setTextureThreshold(10);
			//视差唯一性百分比， 视差窗口范围内最低代价是次低代价的(1 + uniquenessRatio/100)倍时，最低代价对应的视差值才是该像素点的视差，否则该像素点的视差为 0 ，该参数不能为负值，一般5-15左右的值比较合适，int 型
			bm->setUniquenessRatio(rate);
			//检查视差连通区域变化度的窗口大小, 值为 0 时取消 speckle 检查
			bm->setSpeckleWindowSize(100);
			//视差变化阈值，当窗口内视差变化大于阈值时，该窗口内的视差清零
			bm->setSpeckleRange(32);
			//左右视图的有效像素区域，一般由双目校正阶段的 cvStereoRectify 函数传递，也可以自行设定。一旦在状态参数中设定了 roi1 和 roi2，OpenCV 会通过cvGetValidDisparityROI 函数计算出视差图的有效区域，在有效区域外的视差值将被清零。
			bm->setROI1(roi_left);
			bm->setROI2(roi_right);

			int pfs = bm->getPreFilterSize();
			int pfc = bm->getPreFilterCap();
			int bs = bm->getBlockSize();
			int md = bm->getMinDisparity();
			int nd = bm->getNumDisparities();
			int tt = bm->getTextureThreshold();
			int ur = bm->getUniquenessRatio();
			int sw = bm->getSpeckleWindowSize();
			int sr = bm->getSpeckleRange();

			// Compute disparity
			bm->compute(img_left, img_right, disp);
			disp = disp.colRange(80, img_left.cols);
			disp.convertTo(disp, CV_32F, 1.0 / 16);
			cv::imshow("disp", disp);
			// 生成并显示点云
			cv::Mat xyz;
			cv::reprojectImageTo3D(disp, xyz, Q, true);

			saveXYZ("data_disp.txt", xyz);
			cvWaitKey(1);
		}
	}
	pthread_exit(NULL);
}

/**
 * 显示imu数据函数变为动态采集视差唯一性百分比参数
 */
void* show_imuData(void *)
{
	int counter = 0;
	while (1)
	{
		//视差唯一性百分比
		cin >> rate;
		// if (visensor_imu_have_fresh_data())
		// {
		// 	counter++;
		// 	// 每隔20帧显示一次imu数据
		// 	if (counter >= 20)
		// 	{
		// 		float ax = visensor_imudata_pack.ax;
		// 		float ay = visensor_imudata_pack.ay;
		// 		float az = visensor_imudata_pack.az;
		// 		cout << "visensor_imudata_pack->a : " << sqrt(ax * ax + ay * ay + az * az) << endl;
		// 		//cout<<"visensor_imudata_pack->a : "<<visensor_imudata_pack.ax<<" , "<<visensor_imudata_pack.ay<<" , "<<visensor_imudata_pack.az<<endl;
		// 		//cout<<"imu_time1 : "<<visensor_imudata_pack.imu_time<<endl;
		// 		//cout<<"imu_time2 : "<<visensor_imudata_pack.system_time.tv_usec<<endl;
		// 		counter = 0;
		// 	}
		// }
		usleep(50);
	}
	pthread_exit(NULL);
}

int main(int argc, char* argv[])
{

	/************************ Start Cameras ************************/
	visensor_load_settings("Loitor_VISensor_Setups.txt");

	// 手动设置相机参数
	//visensor_set_current_mode(5);
	//visensor_set_auto_EG(0);
	//visensor_set_exposure(50);
	//visensor_set_gain(200);
	//visensor_set_cam_selection_mode(2);
	//visensor_set_resolution(false);
	//visensor_set_fps_mode(true);
	// 保存相机参数到原配置文件
	//visensor_save_current_settings();

	int r = visensor_Start_Cameras();
	if (r < 0)
	{
		printf("Opening cameras failed...\r\n");
		return r;
	}
	/************************** Start IMU **************************/
	int fd = visensor_Start_IMU();
	if (fd < 0)
	{
		printf("visensor_open_port error...\r\n");
		return 0;
	}
	printf("visensor_open_port success...\r\n");
	/************************ ************ ************************/

	usleep(100000);

	//Create img_show thread
	pthread_t showimg_thread;
	int temp;
	if (temp = pthread_create(&showimg_thread, NULL, opencv_showimg, NULL))
		printf("Failed to create thread opencv_showimg\r\n");
	//Create show_imuData thread
	pthread_t showimu_thread;
	if (temp = pthread_create(&showimu_thread, NULL, show_imuData, NULL))
		printf("Failed to create thread show_imuData\r\n");

	while (1)
	{
		// Do - Nothing :)
		//cout<<visensor_get_imu_portname()<<endl;
		//cout<<visensor_get_hardware_fps()<<endl;
		usleep(500000);
	}

	/* shut-down viewers */
	close_img_viewer = true;
	visensor_Close_IMU_viewer = true;
	if (showimg_thread != 0)
	{
		pthread_join(showimg_thread, NULL);
	}
	if (showimu_thread != 0)
	{
		pthread_join(showimu_thread, NULL);
	}

	/* close cameras */
	visensor_Close_Cameras();
	/* close IMU */
	visensor_Close_IMU();

	return 0;
}
