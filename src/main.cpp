#include "tensorrt/engine.h"
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Log.h>
#include <plog/Init.h>
#include <plog/Appenders/ColorConsoleAppender.h>

int main() {
    static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;
    plog::init(plog::info, &consoleAppender);

    std::string onnxModelPath;
    int maxBatchSize;
    int maxSize;

    std::cout << "Enter ONNX model path:";
    std::cin >> onnxModelPath;
    std::cout << "Enter max batch size:";
    std::cin >> maxBatchSize;
    std::cout << "Enter max size:";
    std::cin >> maxSize;

    trt::Config config;
    config.maxBatchSize = maxBatchSize;
    config.maxWidth = maxSize;
    config.maxHeight = maxSize;
    trt::Engine engine(config);
    engine.build(onnxModelPath);
    return 0;
}