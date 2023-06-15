#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <opencv2/opencv.hpp>
#include<sycl/sycl.hpp>

// constexpr float M_PI = 3.141592;// 65354;
using namespace cl::sycl;

// 高斯函数
double gaussian(double dx, int dy, double sigma) {

	float exponent = -(dx * dx + dy * dy) / (2.0f * sigma * sigma);
	return  std::exp(exponent) / (2.0f * M_PI * sigma * sigma);
}
void GPUgaussianBlur(const cv::Mat& input, cv::Mat& output, float sigma, int blockSize) {
	// 创建队列和设备选择器
	gpu_selector selector;
	queue q(selector);

	// 获取输入图像的宽度和高度
	int width = input.rows;
	int height = input.cols;

	// 分配输出图像
	output.create(input.size(), CV_8UC1);

	//分配输入和输出缓冲区
	buffer<uchar, 2> inputBuf(input.ptr<uchar>(), range<2>(width, height));
	buffer<uchar, 2> outputBuf(output.ptr<uchar>(), range<2>(width, height));

	// 提交任务到队列
	q.submit([&](handler& gaussian) {
		// 获取访问器并定义访问范围
		auto inputAccessor = inputBuf.get_access<access::mode::read>(gaussian);
		auto outputAccessor = outputBuf.get_access<access::mode::write>(gaussian);
		auto range = inputBuf.get_range();
		// 定义内核函数
		gaussian.parallel_for(range, [=](id<2> idx) {
			int x = idx[0];
			int y = idx[1];
			const float PI=3.141592;

			//// 计算高斯模糊权重
			auto getWeight = [=](int dx, int dy) {
				float exponent = -(dx * dx + dy * dy) / (2.0f * sigma * sigma);
				return  std::exp(exponent) / (2.0f * PI * sigma * sigma);
			};

			//// 高斯模糊算法的实现
			float blurredPixel = 0.0;
			float totalWeight = 0.0;

			for (int i = -blockSize; i <= blockSize; i++) {
				for (int j = -blockSize; j <= blockSize; j++) {
					int neighborX = x + i;
					int neighborY = y + j;
					if (neighborX >= 0 && neighborX < width && neighborY >= 0 && neighborY < height) {
						float weight = (getWeight(i, j));
						blurredPixel += inputAccessor[neighborX][neighborY] * weight;
						totalWeight += weight;
					}
				}
			}
			outputAccessor[x][y] = static_cast<uchar>(round(blurredPixel / totalWeight));// std::round(blurredPixel / totalWeight);
			});
		});
	q.wait();
}

// 高斯模糊核心函数
void gaussianBlur(const cv::Mat& input, cv::Mat& output, float sigma, int blockSize) {
	// 创建队列和设备选择器
	default_selector selector;
	queue q(selector);

	// 获取输入图像的宽度和高度
	int width = input.rows;
	int height = input.cols;

	// 分配输出图像
	output.create(input.size(), CV_8UC1);

	//分配输入和输出缓冲区
	buffer<uchar, 2> inputBuf(input.ptr<uchar>(), range<2>(width, height));
	buffer<uchar, 2> outputBuf(output.ptr<uchar>(), range<2>(width, height));

	// 提交任务到队列
	q.submit([&](handler& gaussian) {
		// 获取访问器并定义访问范围
		auto inputAccessor = inputBuf.get_access<access::mode::read>(gaussian);
		auto outputAccessor = outputBuf.get_access<access::mode::write>(gaussian);
		auto range = inputBuf.get_range();
		// 定义内核函数
		gaussian.parallel_for(range, [=](id<2> idx) {
			int x = idx[0];
			int y = idx[1];
			const float PI=3.141592;

			//// 计算高斯模糊权重
			auto getWeight = [=](int dx, int dy) {
				float exponent = -(dx * dx + dy * dy) / (2.0f * sigma * sigma);
				return  std::exp(exponent) / (2.0f * PI * sigma * sigma);
			};

			//// 高斯模糊算法的实现
			float blurredPixel = 0.0;
			float totalWeight = 0.0;

			for (int i = -blockSize; i <= blockSize; i++) {
				for (int j = -blockSize; j <= blockSize; j++) {
					int neighborX = x + i;
					int neighborY = y + j;
					if (neighborX >= 0 && neighborX < width && neighborY >= 0 && neighborY < height) {
						float weight = (getWeight(i, j));
						blurredPixel += inputAccessor[neighborX][neighborY] * weight;
						totalWeight += weight;
					}
				}
			}
			outputAccessor[x][y] = static_cast<uchar>(round(blurredPixel / totalWeight));// std::round(blurredPixel / totalWeight);
			});
		});
	q.wait();
}
void MeanFilater(cv::Mat& src, cv::Mat& dst, int size) {
	const int winsize_2 = size;
	int winsize = winsize_2 * 2 + 1; // 防止为偶数
	const float winsize_num = winsize * winsize;   //(2n+1)*(2n+1)
	const int row = src.rows;   //行
	const int col = src.cols;
	cv::Mat dst_tmp(src.size(), CV_8UC1);   //列
	for (int i = 0; i < row; i++)  //行循环，只计算图像的原有行
	{
		for (int j = 0; j < col; j++)  //列循环，只计算图像的原有列
		{
			float sum = 0.0;
			int num = 0;
			//计算每一个像素点周围矩形区域内所有像素点的加权和
			for (int x = i - size; x <= i + size; x++)
			{
				for (int y = j - size; y <= j - size; y++)
				{
					if (x >= 0 && x < row && y >= 0 && y < col)
					{
						sum += src.ptr<uchar>(x)[y];
						num++;
					}
				}
			}
			// 权重和为1， 此处不用除以区域面积
			if(num)
				dst_tmp.ptr<uchar>(i)[j] = (uchar)(sum / num);
		}
	}
	dst_tmp.copyTo(dst);
}

void meanBlur(const cv::Mat& input, cv::Mat& output, float sigma, int blockSize) {
	// 创建队列和设备选择器
	default_selector selector;
	queue q(selector);

	// 获取输入图像的宽度和高度
	int width = input.rows;
	int height = input.cols;

	// 分配输出图像
	output.create(input.size(), CV_8UC1);

	//分配输入和输出缓冲区
	buffer<uchar, 2> inputBuf(input.ptr<uchar>(), range<2>(width, height));
	buffer<uchar, 2> outputBuf(output.ptr<uchar>(), range<2>(width, height));


	// 提交任务到队列
	q.submit([&](handler& mean) {
		//hi
		// 获取访问器并定义访问范围
		auto inputAccessor = inputBuf.get_access<access::mode::read>(mean);
		auto outputAccessor = outputBuf.get_access<access::mode::write>(mean);
		auto range = inputBuf.get_range();
		// 定义内核函数
		mean.parallel_for<class mean_blur>(range, [=](id<2> idx) {
			int x = idx[0];
			int y = idx[1];

			// 返回均值模糊权重
			auto getWeight = [=](int dx, int dy) {
				//float exponent = -(dx * dx + dy * dy) / (2.0f * sigma * sigma);
				//return std::exp(exponent) / (2.0f * M_PI * sigma * sigma);
				return 1;
			};

			// 均值模糊算法的实现
			float blurredPixel = 0.0f;
			float totalWeight = ((blockSize * 2 + 1) * (blockSize * 2 + 1));// 0.0f;

			for (int i = -blockSize; i <= blockSize; i++) {
				for (int j = -blockSize; j <= blockSize; j++) {
					int neighborX = x + i;
					int neighborY = y + j;

					if (neighborX >= 0 && neighborX < width && neighborY >= 0 && neighborY < height) {
						float weight = 1.0f;// getWeight(i, j);
						blurredPixel += inputAccessor[neighborX][neighborY];// *weight;
						//totalWeight += weight;
					}
				}
			}
			outputAccessor[x][y] = static_cast<uchar>(round(blurredPixel / totalWeight));// std::round(blurredPixel / totalWeight);

			});
		});
	q.wait();
}
void meanBlurPlus(const cv::Mat& input, cv::Mat& output, float sigma, int blockSize) {

	// 创建队列和设备选择器
	default_selector selector;
	queue q(selector);

	// 获取输入图像的宽度和高度
	const int rows = input.rows;
	const int cols = input.cols;

	// 分配输出图像
	output.create(input.size(), CV_8UC1);
	//std::cout << rows << " " << cols << "\n";

	//申请共享内存
	int* sharedMemory = malloc_shared<int>(rows * cols, q);

	//分配输入和输出缓冲区
	buffer<uchar, 2> inputBuf(input.ptr<uchar>(), range<2>(rows, cols));
	buffer<uchar, 2> outputBuf(output.ptr<uchar>(), range<2>(rows, cols));

	//行分裂相加
	q.submit([&](handler& add_1) {
		auto inputAccessor = inputBuf.get_access<access::mode::read>(add_1);
		range<2>r(rows, cols / 4);
		add_1.parallel_for(r, [=](id<2>idx) {
			int r = idx[0];
			int c = idx[1] * 4;
			int sharedIdx = r * cols + c;
			sharedMemory[sharedIdx] = inputAccessor[r][c];
			sharedMemory[sharedIdx + 1] = sharedMemory[sharedIdx] + inputAccessor[r][c + 1];
			sharedMemory[sharedIdx + 2] = sharedMemory[sharedIdx + 1] + inputAccessor[r][c + 2];
			sharedMemory[sharedIdx + 3] = sharedMemory[sharedIdx + 2] + inputAccessor[r][c + 3];
			});
		});// .wait();
	//行块末尾求值
	q.submit([&](handler& add_2) {
		auto inputAccessor = inputBuf.get_access<access::mode::read>(add_2);
		range<1>r(rows);
		add_2.parallel_for(r, [=](id<1>idx) {
			int r = idx[0];
			for (int i = 7; i < cols; i += 4) {
				sharedMemory[r * cols + i] += sharedMemory[r * cols + i - 4];
			}
			});
		});// .wait();
	//行块内部求值
	q.submit([&](handler& add_3) {
		auto inputAccessor = inputBuf.get_access<access::mode::read>(add_3);
		range<2>r(rows, cols / 4);
		add_3.parallel_for(r, [=](id<2>idx) {
			int r = idx[0];
			int c = idx[1] * 4;
			int sharedIdx = r * cols + c;
			if (c > 0)
			{
				sharedMemory[sharedIdx] += sharedMemory[sharedIdx - 1];
				sharedMemory[sharedIdx + 1] += sharedMemory[sharedIdx - 1];
				sharedMemory[sharedIdx + 2] += sharedMemory[sharedIdx - 1];
			}
			});
		});// .wait();
	//列分裂相加
	q.submit([&](handler& add_4) {
		auto inputAccessor = inputBuf.get_access<access::mode::read>(add_4);
		range<2>r(cols, rows / 4);
		add_4.parallel_for(r, [=](id<2>idx) {
			int c = idx[0];
			int r = idx[1] * 4;
			int sharedIdx = r * cols + c;
			sharedMemory[sharedIdx] = sharedMemory[sharedIdx];// inputAccessor[r][c];
			sharedMemory[sharedIdx + cols] += sharedMemory[sharedIdx];// inputAccessor[r][c + 1];
			sharedMemory[sharedIdx + cols * 2] += sharedMemory[sharedIdx + cols * 1];// inputAccessor[r][c + 2];
			sharedMemory[sharedIdx + cols * 3] += sharedMemory[sharedIdx + cols * 2];// inputAccessor[r][c + 3];
			});
		});// .wait();
	//列块末尾求值
	q.submit([&](handler& add_5) {
		auto inputAccessor = inputBuf.get_access<access::mode::read>(add_5);
		range<1>r(cols);
		add_5.parallel_for(r, [=](id<1>idx) {
			int c = idx[0];
			for (int i = 7; i < rows; i += 4) {
				sharedMemory[i * cols + c] += sharedMemory[cols * (i - 4) + c];
			}
			});
		});
	//列块内部求值
	q.submit([&](handler& add_6) {
		auto inputAccessor = inputBuf.get_access<access::mode::read>(add_6);
		range<2>r(rows / 4, cols);
		add_6.parallel_for(r, [=](id<2>idx) {
			int r = idx[0] * 4;
			int c = idx[1];
			int sharedIdx = r * cols + c;
			if (r > 0)
			{
				sharedMemory[sharedIdx] += sharedMemory[sharedIdx - cols];
				sharedMemory[sharedIdx + cols] += sharedMemory[sharedIdx - cols];
				sharedMemory[sharedIdx + cols * 2] += sharedMemory[sharedIdx - cols];
			}
			});
		}).wait();
		// 提交任务到队列
		q.submit([&](handler& mean) {
			// 获取访问器并定义访问范围
			auto inputAccessor = inputBuf.get_access<access::mode::read>(mean);
			auto outputAccessor = outputBuf.get_access<access::mode::write>(mean);
			//auto range = inputBuf.get_range();
			range<2>r(rows, cols);
			//定义内核函数
			mean.parallel_for(r, [=](id<2> idx) {
				int x = idx[0];
				int y = idx[1];
				// 返回坐标值
				auto getId = [=](int dx, int dy) {
					return dx * cols + dy;
				};
				int lx = x - blockSize;
				int ly = y - blockSize;
				int rx = x + blockSize;
				int ry = y + blockSize;
				float totalWeight = ((blockSize * 2 + 1) * (blockSize * 2 + 1));
				float blurredPixel = 0.0;
				if (lx <= 0 || ly <= 0)
					if (rx < rows && ry < cols)
						blurredPixel = inputAccessor[x][y]*totalWeight;
					else
						blurredPixel = inputAccessor[x][y]*totalWeight;
						//blurredPixel = totalWeight;
				else
				{
					rx = min(rows-1, rx);
					ry = min(cols-1, ry);
					blurredPixel = sharedMemory[getId(rx, ry)] + sharedMemory[getId(lx - 1, ly - 1)] - sharedMemory[getId(lx - 1, ry)] - sharedMemory[getId(rx, ly - 1)];
                    // if(rx>=rows||ry>=rows);
                    //     totalWeight=(rx-)
                }
				//outputAccessor[x][y] = static_cast<uchar>(round(1));// std::round(blurredPixel / totalWeight);
				outputAccessor[x][y] = static_cast<uchar>((blurredPixel / totalWeight));// std::round(blurredPixel / totalWeight);
				});
			});
		q.wait();
		//std::cout << round(1) << std::endl;
}
void GaussianFilter(const cv::Mat& src, cv::Mat& dst, int winsize, double sigma) {
	const int winsize_2 = winsize / 2;
	winsize = winsize_2 * 2 + 1; // 防止为偶数
	const float winsize_num = winsize * winsize;   //(2n+1)*(2n+1)
	cv::Mat src_board;
	copyMakeBorder(src, src_board, winsize_2, winsize_2, winsize_2, winsize_2, cv::BORDER_REFLECT);

	const int row = src_board.rows;   //行
	const int col = src_board.cols;
	cv::Mat dst_tmp(src.size(), CV_8UC1);   //列
	cv::Mat kernel = cv::getGaussianKernel(winsize, sigma);
	kernel = kernel * kernel.t();
	for (int i = winsize_2; i < row - winsize_2; i++)  //行循环，只计算图像的原有行
	{
		for (int j = winsize_2; j < col - winsize_2; j++)  //列循环，只计算图像的原有列
		{
			float sum = 0.0;
			//计算每一个像素点周围矩形区域内所有像素点的加权和
			for (int y = 0; y < winsize; y++)
			{
				for (int x = 0; x < winsize; x++)
				{
					sum += src_board.ptr<uchar>(i - winsize_2 + y)[j - winsize_2 + x] * kernel.at<double>(y, x);
				}
			}
			// 权重和为1， 此处不用除以区域面积
			dst_tmp.ptr<uchar>(i - winsize_2)[j - winsize_2] = (uchar)(sum + 0.5);
		}
	}
	dst_tmp.copyTo(dst);
}
/*
./main gaussian.png
./main images.jpg
./main lena.png
*/
int main(int argc,char *argv[])
 {
	printf("%s\n",argv[1]);
	const char* inputFilename = argv[1];//"gaussian.png";
	const char* gaussianOutFilename = "gaussianOut.jpg";
	const char* meanOutFilename = "meanOut.jpg";
	const char* meanPlusOutFilename = "meanPlusOut.jpg";
	// freopen("data.txt","w",stdout);
	float sigma;
	int blockSize=10;
	//1080 720 540 
	//for(int sizz=90;sizz<=1080;sizz+=90)
	{
	//	for(int blockSize=1;(2*blockSize+1)<sizz;blockSize=blockSize*2)
		{
			scanf("%d",&blockSize);
			sigma=(1.0*blockSize*2+1)/3;
			//printf("%d %d ",sizz,blockSize);
			clock_t start_t, end_t,t1,t2,t3,t4,t5;
			double total_t;
			int i;
			start_t = clock();
			// 读取输入图像
			// cv::Mat input(sizz/9*16,sizz,CV_8UC1);
			cv::Mat input = cv::imread(inputFilename, cv::IMREAD_GRAYSCALE);
			//使图像正规化
			if ((input.cols % 4) || (input.rows % 4))
			{
				cv::copyMakeBorder(input, input, 0, 4 - (input.rows % 4), 0, 4 - (input.cols % 4), CV_HAL_BORDER_REFLECT);
			}
			// if (input.empty()) {
			// 	std::cerr << "Failed to open image file: " << inputFilename << std::endl;
			// 	return 1;
			// }
			// 分配输出图像
			cv::Mat gaussianOut(input.size(), CV_8UC1);
			cv::Mat meanPlusOut(input.size(), CV_8UC1);
			cv::Mat meanOut(input.size(), CV_8UC1);

			cv::Mat dst2;


			t1=clock();
			// 调用均值模糊函数
			meanBlur(input, meanOut, sigma, blockSize);
			t2=clock();
			//调用高斯模糊函数
			//MeanFilater(input,meanOut,blockSize);
			GaussianFilter(input,gaussianOut,blockSize,sigma);
				// gaussianBlur(input, gaussianOut, sigma, blockSize);
			//cv::GaussianBlur(input, dst2, cv::Size(blockSize, blockSize) , 0);
			t3=clock();
			//调用前置和优化均值滤波
			meanBlurPlus(input, meanPlusOut, sigma, blockSize);
			t4=clock();
			gaussianBlur(input, gaussianOut, sigma, blockSize);
			t5=clock();
	//		std::cout<<(double)(t2-t1)/ CLOCKS_PER_SEC<<" "<<(double)(t3-t2)/ CLOCKS_PER_SEC<<" "<<(double)(t4-t3)/ CLOCKS_PER_SEC<<" "<<(double)(t5-t4)/ CLOCKS_PER_SEC<<std::endl;
			std::cout<<"mean time:"<<(double)(t2 - t1) / CLOCKS_PER_SEC<<std::endl;
			std::cout<<"gaussian time:"<<(double)(t3 - t2) / CLOCKS_PER_SEC<<std::endl;
			std::cout<<"meanPlus time:"<<(double)(t4 - t3) / CLOCKS_PER_SEC<<std::endl;
			std::cout<<"GPUgaussian time:"<<(double)(t5 - t4) / CLOCKS_PER_SEC<<std::endl;

			// 保存输出图像
			if (!cv::imwrite(gaussianOutFilename, gaussianOut)) {
				std::cerr << "Failed to save image file: " << gaussianOutFilename << std::endl;
				return 1;
			}
			// 保存输出图像
			if (!cv::imwrite(meanOutFilename, meanOut)) {
				std::cerr << "Failed to save image file: " << meanOutFilename << std::endl;
				return 1;
			}
			if (!cv::imwrite(meanPlusOutFilename, meanPlusOut)) {
				std::cerr << "Failed to save image file: " << meanPlusOutFilename << std::endl;
				return 1;
			}
			std::cout << "Image processed successfully." << std::endl;			
		}
	}

	// cv::namedWindow("原图", cv::WINDOW_NORMAL);
	// cv::imshow("原图", input);
	// cv::namedWindow("高斯滤波", cv::WINDOW_NORMAL);
	// cv::imshow("高斯滤波", gaussianOut);
	// cv::namedWindow("前置和优化均值滤波", cv::WINDOW_NORMAL);
	// cv::imshow("前置和优化均值滤波", meanPlusOut);
	// cv::namedWindow("均值滤波", cv::WINDOW_NORMAL);
	// cv::imshow("均值滤波", meanOut);
	// cv::waitKey(0);

	return 0;
}