#include "tensorrt/img2img.h"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Log.h>
#include <plog/Init.h>
#include <plog/Appenders/ColorConsoleAppender.h>

int main() {
    static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;
    plog::init(plog::info, &consoleAppender);

    std::string imagePath;
    std::string modelPath;
    int batchSize;
    int tileSize;

    std::cout << "Enter image path:";
    std::cin >> imagePath;
    std::cout << "Enter engine path:";
    std::cin >> modelPath;
    std::cout << "Enter batch tileSize:";
    std::cin >> batchSize;
    std::cout << "Enter tileSize:";
    std::cin >> tileSize;

    trt::InferrerConfig inferConfig;
    inferConfig.deviceId = 0;
    inferConfig.precision = trt::Precision::FP16;
    inferConfig.inputShape = nvinfer1::Dims4{1, 3, tileSize, tileSize};

    trt::Img2Img engine;
    engine.load(modelPath, inferConfig);

    cv::Mat image = cv::imread(imagePath);
    cv::cuda::GpuMat input;
    input.upload(image);
    cv::cuda::cvtColor(input, input, cv::COLOR_BGR2RGB);
    std::vector<cv::cuda::GpuMat> inputs { input };
    std::vector<cv::cuda::GpuMat> outputs;

    engine.infer(inputs, outputs);

    cv::Mat output;
    outputs[0].download(output);
    cv::cvtColor(output, output, cv::COLOR_RGB2BGR);
    cv::imshow("output", output);
    cv::waitKey(-1);

    return 0;
}