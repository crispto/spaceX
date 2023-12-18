#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/core/types.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <string>
#include <thread>
#include <gperftools/profiler.h>

using namespace std;
using namespace cv;
int main(int argc, char **argv)
{
    ProfilerStart("profile_capture.prof");
    int i =0;
    int count = 0;
    std::cout <<" argc " <<argc <<std::endl;
    if (argc > 1){
        count = atoi(argv[1]);
    }
    if (count <=0){
        std::cout <<"count invalid, use 10000 " << std::endl;
        count = 10000;
    }
    std::cout <<"count is " << count << std::endl;
    while(i < count){
    i++;
    cv::Mat m = cv::imread("data/crowd.jpeg");
    cv::Mat out;
    cv::resize(m, out, cv::Size(), 0.5,0.5, cv::INTER_LINEAR);
    cout <<"Mat size" << out.size << endl;
    cv::imwrite("scale_with_xy.jpeg", out);

    cv::Size resized_shape{600, 400};
    cv::Mat out2;
    cv::resize(m, out2, resized_shape, 0,0, cv::INTER_LINEAR);
    cv::imwrite("scale_by_size.jpeg", out);
    // this_thread::sleep_for(chrono::milliseconds(100));
    }
    ProfilerStop();
    std::cout <<"profiling end" << std::endl;

    return 0;
}
