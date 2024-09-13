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
    plog::init(plog::debug, &consoleAppender);

    std::string imageDir = R"(C:\waifu2x-tensorrt\images\)";
    std::string imagePath = imageDir + "bg.png";
    std::string modelPath = R"(C:\waifu2x-tensorrt\models\swin_unet\art\noise3_scale4x.NVIDIAGeForceRTX3060Ti.FP16.1.1.1.64.256.256.64.256.256.trt)";
    int batchSize = 1;
    int tileSize = 256;

    trt::InferrerConfig inferConfig;
    inferConfig.deviceId = 0;
    inferConfig.precision = trt::Precision::FP16;
    inferConfig.nbBatches = batchSize;
    inferConfig.channels = 3;
    inferConfig.height = tileSize;
    inferConfig.width = tileSize;

    trt::Img2Img engine;
    engine.load(modelPath, inferConfig);

    cv::Mat input = cv::imread(imagePath);
    cv::Mat output;
    cv::cuda::GpuMat gpuInput;
    cv::cuda::GpuMat gpuOutput;
    gpuInput.upload(input);
    cv::cuda::cvtColor(gpuInput, gpuInput, cv::COLOR_BGR2RGB);

    engine.process(gpuInput, gpuOutput, cv::Point2i(4, 4), cv::Point2f(0.125f, 0.125f));
    cv::TickMeter tm;
    for (int i = 0; i < 10; ++i) {
        tm.start();
        engine.process(gpuInput, gpuOutput, cv::Point2i(4, 4), cv::Point2f(0.125f, 0.125f));
        tm.stop();
    }
    std::cout << "Average time: " << tm.getTimeMilli() / 10 << std::endl;

    cv::cuda::cvtColor(gpuOutput, gpuOutput, cv::COLOR_RGB2BGR);
    gpuOutput.download(output);
    cv::imwrite(imageDir + "out.png", output);

    return 0;
}