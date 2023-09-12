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
    std::string modelPath = R"(C:\waifu2x-tensorrt\models\swin_unet\art\noise1_scale4x.NVIDIAGeForceRTX3060Ti.FP16.1.1.1.256.256.256.256.256.256.trt)";
    int deviceId = 0;
    trt::Precision precision = trt::Precision::FP16;
    int batchSize = 1;
    int tileSize = 256;
    cv::Point2i scaling(4, 4);
    cv::Point2d overlap(0.125f, 0.125f);

    int iterations = 100;

    trt::RenderConfig renderConfig;
    renderConfig.deviceId = deviceId;
    renderConfig.precision = precision;
    renderConfig.nbBatches = batchSize;
    renderConfig.channels = 3;
    renderConfig.height = tileSize;
    renderConfig.width = tileSize;
    renderConfig.scaling = scaling;
    renderConfig.overlap = overlap;

    trt::Img2Img engine;
    engine.load(modelPath, renderConfig);

    cv::Mat input = cv::imread(imagePath);
    cv::Mat output;
    cv::cuda::GpuMat gpuInput;
    cv::cuda::GpuMat gpuOutput;
    gpuInput.upload(input);
    cv::cuda::cvtColor(gpuInput, gpuInput, cv::COLOR_BGR2RGB);

    engine.render(gpuInput, gpuOutput);
    cv::TickMeter tm;
    for (int i = 0; i < iterations; ++i) {
        tm.start();
        engine.render(gpuInput, gpuOutput);
        tm.stop();
    }
    std::cout << "Total time: " << tm.getTimeMilli() << std::endl;
    std::cout << "Average time: " << tm.getTimeMilli() / iterations << std::endl;

    cv::cuda::cvtColor(gpuOutput, gpuOutput, cv::COLOR_RGB2BGR);
    gpuOutput.download(output);
    cv::imwrite(imageDir + "out.png", output);

    system("pause");

    return 0;
}