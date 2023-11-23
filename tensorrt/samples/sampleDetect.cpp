#include "command_line_parser.hpp"
#include "common.h"
#include "logger.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <co/cout.h>
#include <co/fastring.h>
#include <co/fs.h>
#include <co/path.h>
#include <cstddef>
#include <fstream>
#include <memory>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <sstream>
using namespace std;
using namespace nvinfer1;
// 一个 struct 可以作为模板参数
struct InferDeleter {
    template <typename T> void operator()(T *obj) const
    {
        delete obj;
    }
};
fastring filename(fastring &&p)
{
    auto pos = p.rfind('/');
    if (pos == fastring::npos) {
        return p;
    }
    return p.substr(pos + 1);
}
fastring remove_ext(fastring &filename)
{
    auto pos = filename.rfind('.');
    if (pos == fastring::npos) {
        return filename;
    }
    return filename.substr(0, pos);
}
void transpose(cv::Mat &hwcMat, std::shared_ptr<float> input_blob_s)
{
    float *input_blob = input_blob_s.get();
    int height = hwcMat.rows;
    int width = hwcMat.cols;
    int channel = hwcMat.channels();
    for (int i = 0; i < height; i++) {
        uchar *data = hwcMat.ptr<uchar>(i);
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < channel; k++) {
                input_blob[k * height * width + i * width + j] = data[j * channel + k] / 255.0;
            }
        }
    }
}

template <typename T> std::vector<size_t> argsort(std::vector<T> &array, bool less(T, T))
{
    std::vector<size_t> indices(array.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&array, &less](int left, int right) -> bool {
        // sort indices according to corresponding array element
        return less(array[left], array[right]);
    });

    return indices;
}
/**
 * @brief nms为不同的类做做最大值抑制
 *
 * @param bbox 所有的 bbox
 * @param width 每个 bbox 的宽度
 * @param height bbox 数量
 * @param threshold
 * @return std::vector<int>
 */
std::vector<float *> nms_yolo(float *bbox, const int width, const int height, const float threshold = 0.3,
                              const float iou_thresh = 0.45)
{

    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        if (x1min > x2min) {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }
        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };

    auto computeIoU = [&overlap1D](float *bbox1, float *bbox2) -> float {
        float overlapX = overlap1D(bbox1[0], bbox1[2], bbox2[0], bbox2[2]);
        float overlapY = overlap1D(bbox1[1], bbox1[3], bbox2[1], bbox2[3]);
        float area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
        float area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };
    // sort indices by score

    // 找到每一行的最大分数值和索引
    std::vector<std::pair<int, float>> label_score;
    for (int i = 0; i < height; i++) {
        float max_score = 0;
        int max_index = 0;

        for (int j = max_index + 4; j < width; j++) {
            if (bbox[i * width + j] > max_score) {
                max_score = bbox[i * width + j];
                max_index = j - 4;
            }
        }
        label_score.push_back(std::make_pair(max_index, max_score));
    }
    // 先按分数过滤所有待处理 bbox
    std::vector<float *> good_bboxes;
    std::vector<std::pair<int, float>> good_label_score;
    for (int i = 0; i < height; i++) {
        if (label_score[i].second > threshold) {
            good_bboxes.push_back(bbox + i * width);
            good_label_score.push_back(label_score[i]);
        }
    }

    co::print("good bboxes size: ", good_bboxes.size());
    // 找到所有的待处理 bbox
    // cv::Mat data(cv::Mat_<float>(good_bboxes.size(), 6)); //[bbox, label, score]
    cv::Mat data(good_bboxes.size(), 6, CV_32FC1); //[bbox, label, score]
    for (int i = 0; i < good_bboxes.size(); i++) {
        // convert center width height to x1 y1 x2 y2
        data.at<float>(i, 0) = good_bboxes[i][0] - good_bboxes[i][2] / 2;
        data.at<float>(i, 1) = good_bboxes[i][1] - good_bboxes[i][3] / 2;
        data.at<float>(i, 2) = good_bboxes[i][0] + good_bboxes[i][2] / 2;
        data.at<float>(i, 3) = good_bboxes[i][1] + good_bboxes[i][3] / 2;
        data.at<float>(i, 4) = float(good_label_score[i].first);
        data.at<float>(i, 5) = good_label_score[i].second;
    }
    cout << "good bbox " << data << endl;
    // 所有的检测结果类别
    std::set<float> all_labels;
    for (int i = 0; i < good_bboxes.size(); i++) {
        all_labels.insert(data.at<float>(i, 4));
    }
    std::vector<float> all_label_vec{ all_labels.begin(), all_labels.end() };
    std::cout << "size of all labels " << all_label_vec.size() << endl;
    // print all labels:
    for (auto label : all_label_vec) {
        std::cout << label << " ," << endl;
    }
    std::vector<float *> output_bboxes;
    // 对每一类做nms
    for (auto label : all_label_vec) {
        std::vector<float *> indices;
        for (int i = 0; i < good_bboxes.size(); i++) {
            if (data.at<float>(i, 4) == label) {
                indices.push_back(good_bboxes[i]);
            }
        }
        // 对得分进行倒序排序
        std::vector<size_t> sorted_indices = argsort<float *>(indices, [](float *a, float *b) -> bool { return a[5] > b[5]; });

        // do nms
        std::cout << " size of sorted_indices: " << sorted_indices.size() << endl;
        std::vector<size_t> keep; //保存下一轮的临时值
        while (sorted_indices.size() > 0) {
            keep.clear();
            float *M = data.ptr<float>(sorted_indices[0]); // 最大 bbox
            output_bboxes.push_back(M);
            for (int i = 1; i < sorted_indices.size(); i++) {
                float *N = data.ptr<float>(sorted_indices[i]);
                // x1, x2, y1, y2 格式 
                if (computeIoU(M, N) < iou_thresh) {
                    keep.push_back(sorted_indices[i]);
                }
            }
            std::swap(sorted_indices, keep);
        }
    }
    return output_bboxes;

    // 按照每一种类别做nms
}

// 可视化
void process_output(cv::Mat &img, std::shared_ptr<float> output_blob, int height, int width)
{
    static vector<string> label_name = {
        "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
        "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
        "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
        "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
        "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
        "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
        "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
        "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
        "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
        "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
        "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
        "teddy bear",     "hair drier", "toothbrush"
    };

    // run nms
    float *data = output_blob.get();
    std::vector<float *> bboxes = nms_yolo(data, width, height, 0.5);
    cout << "total bboxe after nms_yolo : " << bboxes.size() << endl;
    float x_scale = img.cols / 640.0;
    float y_scale = img.rows / 640.0;
    cout << "x_scale: " << x_scale << " y_scale: " << y_scale << endl;
    for (auto bbox : bboxes) {
        char buf[20];
        cv::Rect rect(bbox[0] * x_scale, bbox[1] * y_scale, (bbox[2] - bbox[0]) * x_scale, (bbox[3] - bbox[1]) * y_scale);
        cout << "rect is " << rect << endl;
        cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 2);
        sprintf(buf, "%s: %.2f", label_name[int(bbox[4])].c_str(), bbox[5]);
        cv::putText(img, buf, cv::Point(rect.x, rect.y - 14), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 2);
    }
    cv::namedWindow("default", cv::WINDOW_NORMAL);
    cv::resizeWindow("default", cv::Size(img.cols, img.rows));
    cv::imshow("default", img);
    cv::waitKey(-1);
    cv::imwrite("result.jpg", img);
}
void run_inference(string image_path, unique_ptr<ICudaEngine> &engine, unique_ptr<IRuntime> &runtime)
{
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        sample::gLogError << "read image failed" << endl;
        exit(EXIT_FAILURE);
    }
    int width = 640;
    int height = 640;
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(width, height));
    //
    std::shared_ptr<float> input_blob(new float[3 * width * height], [](float *p) { delete[] p; });
    transpose(resized_img, input_blob);

    // print input_blob
    float *start = input_blob.get();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 100; j++) {
            std::cout << start[i * width * height + j] << " ,";
        }
        cout << endl;
    }

    // input device memory
    nvinfer1::Dims input_dim = engine->getBindingDimensions(0);
    size_t input_size = 1;
    for (size_t i = 0; i < input_dim.nbDims; i++) {
        input_size *= input_dim.d[i];
    }
    void *device_buffer[2];
    cudaMalloc(&device_buffer[0], input_size * sizeof(float));

    // output device_memory
    nvinfer1::Dims output_dim = engine->getBindingDimensions(1);
    size_t output_size = 1;
    for (size_t i = 0; i < output_dim.nbDims; i++) {
        output_size *= output_dim.d[i];
    }
    cudaMalloc(&device_buffer[1], output_size * sizeof(float));

    std::shared_ptr<float> output_blob(new float[output_size], std::default_delete<float[]>());
    cout << "binding num " << engine->getNbBindings() << endl;
    // 创建推理上下文
    std::unique_ptr<nvinfer1::IExecutionContext> context{ engine->createExecutionContext() };
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudaMemcpyAsync(device_buffer[0], input_blob.get(), input_size * sizeof(float), cudaMemcpyHostToDevice, stream);
    // 这里跑了推理
    context->enqueueV2(device_buffer, stream, nullptr);
    cudaMemcpyAsync(output_blob.get(), device_buffer[1], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    co::print("input size: ", input_size, " output size: ", output_size);
    cudaFree(device_buffer[0]);
    cudaFree(device_buffer[1]);
    cudaStreamDestroy(stream);

    int outputDim1 = output_dim.d[1]; // 84
    int outputDim2 = output_dim.d[2]; // 8400

    // 这样做有些愚蠢
    shared_ptr<float> transposed{ new float[output_size], std::default_delete<float[]>() };
    for (int i = 0; i < outputDim1; i++) {
        for (int j = 0; j < outputDim2; j++) {
            transposed.get()[j * outputDim1 + i] = output_blob.get()[i * outputDim2 + j];
        }
    }

    cout << "output dim: " << outputDim1 << " " << outputDim2 << endl;
    // post process
    process_output(img, transposed, outputDim2, outputDim1);
    return;
}

int main(int argc, char **argv)
{
    auto argparser = CommandLineParser(argv[0], "TensorRT sample for object detection");
    argparser.AddArgument("input_path", "输入图片地址");
    argparser.AddOption("--model_path", 1, "-m", "模型地址: onnx");
    argparser.AddOption("--engine_path", 1, "-e", "engine 保存地址");
    argparser.Parse(argc, argv);

    string input_path = argparser["input_path"].ToString();
    if (!fs::exists(input_path)) {
        sample::gLogError << "input_path not exists" << endl;
        exit(EXIT_FAILURE);
    }
    string engine_path = argparser["-e"].ToString();
    string model_path = argparser["-m"].ToString();
    // check path exists
    co::print("engine_path: ", engine_path);
    co::print("model_path: ", model_path);

    if (!model_path.empty() && !fs::exists(model_path)) {
        sample::gLogError << "model_path not exists" << endl;
        exit(EXIT_FAILURE);
    }
    bool load_engine = !engine_path.empty(); // 直接加载 engine
    if (load_engine && !fs::exists(engine_path)) {
        sample::gLogError << "engine_path not exists" << endl;
        exit(EXIT_FAILURE);
    }

    // 模式1： 直接加载之前 dump 的 engine
    if (load_engine) {

        // load engine
        ifstream engine_file(engine_path.c_str(), ios::binary);
        engine_file.seekg(0, ios::end);
        size_t size = engine_file.tellg();
        engine_file.seekg(0, ios::beg);
        unique_ptr<char[]> engine_data(new char[size]);
        engine_file.read(engine_data.get(), size);
        unique_ptr<IRuntime> runtime{ createInferRuntime(sample::gLogger) };
        unique_ptr<ICudaEngine> engine{ runtime->deserializeCudaEngine(engine_data.get(), size) };
        if (!engine) {
            sample::gLogError << "Build Engine failed" << endl;
            exit(EXIT_FAILURE);
        }
        // show dims
        ASSERT(engine->getNbBindings() == 2);
        Dims mInputDims = engine->getBindingDimensions(0);
        Dims mOutputDims = engine->getBindingDimensions(1);
        co::print("input dims: ", mInputDims.nbDims);
        co::print("output dims: ", mOutputDims.nbDims);
        co::print("binds nums: ", engine->getNbBindings());
        run_inference(input_path, engine, runtime);
        // run infer
    } else {
        // 模式二： 重新加载 onnx 模型，然后 dump engine
        // write engine plan to file
        fastring modelname = filename(fastring(model_path));

        auto engine_plan_path = path::join(path::dir(model_path), remove_ext(modelname).cat(".engine"));
        sample::gLogInfo << "write engine plan to file: " << engine_plan_path;

        // builder
        auto builder = unique_ptr<IBuilder>(createInferBuilder(sample::gLogger));

        // netork
        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = unique_ptr<INetworkDefinition>(builder->createNetworkV2(explicitBatch));

        // parser， 这里 parser 会代理network
        auto parser = unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger));
        // config
        auto config = unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
        config->setMaxWorkspaceSize(2000_MiB);
        config->setFlag(BuilderFlag::kFP16);
        // 这里会对 network 进行修改
        auto success = parser->parseFromFile(model_path.c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));

        if (!success) {
            sample::gLogError << "Failed to parse onnx file" << endl;
            exit(EXIT_FAILURE);
        }

        // 产生序列化的 engine plane 并保存到文件
        unique_ptr<IHostMemory> engine_plan{ builder->buildSerializedNetwork(*network, *config) };

        ofstream engine_plan_file(engine_plan_path.c_str(), ios::binary);

        engine_plan_file.write(static_cast<const char *>(engine_plan->data()), engine_plan->size());
        engine_plan_file.close();
        // 从engine_plan中加载 engine plane
        unique_ptr<IRuntime> runtime{ createInferRuntime(sample::gLogger) };
        unique_ptr<ICudaEngine> engine{ runtime->deserializeCudaEngine(engine_plan->data(), engine_plan->size()) };
        if (!engine) {
            sample::gLogError << "Build Engine failed" << endl;
            exit(EXIT_FAILURE);
        }

        ASSERT(network->getNbInputs() == 1);
        Dims mInputDims = network->getInput(0)->getDimensions();

        ASSERT(network->getNbOutputs() == 1);
        Dims mOutputDims = network->getOutput(0)->getDimensions();

        // show dims
        co::print("input dims: ", mInputDims.nbDims);
        co::print("output dims: ", mOutputDims.nbDims);
        co::print("binds nums: ", engine->getNbBindings());
    }

    // run infer

    return 0;
}
