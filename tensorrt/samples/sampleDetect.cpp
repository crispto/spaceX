#include "command_line_parser.hpp"
#include "common.h"
#include "logger.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <co/cout.h>
#include <co/fastring.h>
#include <co/fs.h>
#include <co/path.h>
#include <fstream>
#include <memory>
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

int main(int argc, char **argv)
{
    auto argparser = CommandLineParser(argv[0], "TensorRT sample for object detection");
    argparser.AddOption("--model_path", 1, "-m", "模型地址: onnx");
    argparser.AddOption("--engine_path", 1, "-e", "engine 保存地址");
    argparser.Parse(argc, argv);
    string engine_path = argparser["-e"].ToString();
    string model_path = argparser["-m"].ToString();
    if (engine_path.empty() && model_path.empty()) {
        sample::gLogError << "engine_path and model_path can not be empty" << endl;
        exit(EXIT_FAILURE);
    }
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
