#include "tensorrt/engine.h"
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Log.h>
#include <plog/Init.h>
#include <plog/Appenders/ColorConsoleAppender.h>

int main() {
    static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;
    plog::init(plog::info, &consoleAppender);

    std::string modelPath;
    int batchSize;
    int size;

    std::cout << "Enter engine path:";
    std::cin >> modelPath;
    std::cout << "Enter batch size:";
    std::cin >> batchSize;
    std::cout << "Enter size:";
    std::cin >> size;

    trt::BuilderConfig config;
    config.maxBatchSize = batchSize;
    config.maxWidth = size;
    config.maxHeight = size;

    trt::InferrerConfig inferConfig;
    inferConfig.deviceId = 0;
    inferConfig.precision = trt::Precision::FP16;
    inferConfig.inputShape = nvinfer1::Dims4{1, 3, size, size};

    trt::SuperResEngine engine(config);
    engine.load(modelPath, inferConfig);
    //engine.build(modelPath);
    return 0;
}