#include <iostream>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "tensorrt/img2img.h"

#define SPDLOG_LEVEL_NAMES { "TRACE", "DEBUG", "INFO ", "WARN ", "ERROR", "FATAL", "OFF" }
#include <spdlog/sinks/stdout_color_sinks.h>

int main() {
    auto console = spdlog::stdout_color_mt("console");
    console->set_level(spdlog::level::debug);
    console->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");

    trt::LogCallback callback = [&console](trt::Severity severity, const std::string& message, const std::string& file, const std::string& function, int line) {
        std::string s = "[" + function + "@" + std::to_string(line) + "] " + message;
        switch (severity) {
            case trt::Severity::critical:
                console->critical(s);
                break;
            case trt::Severity::error:
                console->error(s);
                break;
            case trt::Severity::warn:
                console->warn(s);
                break;
            case trt::Severity::info:
                console->info(s);
                break;
            case trt::Severity::debug:
                console->debug(s);
                break;
            case trt::Severity::trace:
                console->trace(s);
                break;
        }
    };

    std::string imageDir = R"(C:\waifu2x-tensorrt\images\)";
    std::string imagePath = imageDir + "bg.png";
    std::string modelPath = R"(C:\waifu2x-tensorrt\models\swin_unet\art\noise3_scale4x.onnx)";
    int deviceId = 0;
    trt::Precision precision = trt::Precision::FP16;
    int batchSize = 2;
    int tileSize = 640;
    cv::Point2i scaling(4, 4);
    cv::Point2d overlap(1.0 / 16.0, 1.0 / 16.0);

    int iterations = 1;

    trt::RenderConfig renderConfig;
    renderConfig.deviceId = deviceId;
    renderConfig.precision = precision;
    renderConfig.batchSize = batchSize;
    renderConfig.channels = 3;
    renderConfig.height = tileSize;
    renderConfig.width = tileSize;
    renderConfig.scaling = scaling;
    renderConfig.overlap = overlap;
    renderConfig.tta = true;

    trt::Img2Img engine;
    engine.setLogCallback(callback);
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
    console->info("Total time: {}", tm.getTimeMilli());
    console->info("Average time: {}", tm.getTimeMilli() / iterations);

    gpuOutput.download(output);
    cv::imwrite(imageDir + "out.png", output);

    return 0;
}