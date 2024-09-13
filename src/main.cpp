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

#if false
    std::string modelPath = R"(C:\waifu2x-tensorrt\models\swin_unet\art\noise3_scale4x.onnx)";
    trt::BuildConfig config;
    config.deviceId = 0;
    config.precision = trt::Precision::FP16;
    config.minBatchSize = 1;
    config.optBatchSize = 2;
    config.maxBatchSize = 4;
    config.minChannels = 3;
    config.optChannels = 3;
    config.maxChannels = 3;
    config.minWidth = 256;
    config.optWidth = 256;
    config.maxWidth = 256;
    config.minHeight = 256;
    config.optHeight = 256;
    config.maxHeight = 256;

    trt::Img2Img i;
    i.setLogCallback(callback);
    i.build(modelPath, config);
#else
    std::string imageDir = R"(C:\waifu2x-tensorrt\images\)";
    std::string imagePath = imageDir + "bg.png";
    std::string modelPath = R"(C:\waifu2x-tensorrt\models\swin_unet\art\noise3_scale4x.onnx)";
    int deviceId = 0;
    trt::Precision precision = trt::Precision::FP16;
    int batchSize = 2;
    int tileSize = 256;
    int scaling = 4;
    cv::Point2d overlap(1.0 / 16.0, 1.0 / 16.0);

    int iterations = 1;

    trt::RenderConfig renderConfig1;
    renderConfig1.deviceId = deviceId;
    renderConfig1.precision = precision;
    renderConfig1.batchSize = 4;
    renderConfig1.channels = 3;
    renderConfig1.height = tileSize;
    renderConfig1.width = tileSize;
    renderConfig1.scaling = scaling;
    renderConfig1.overlap = cv::Point2d(0, 0);
    renderConfig1.tta = false;

    trt::RenderConfig renderConfig0;
    renderConfig0.deviceId = deviceId;
    renderConfig0.precision = precision;
    renderConfig0.batchSize = batchSize;
    renderConfig0.channels = 3;
    renderConfig0.height = tileSize;
    renderConfig0.width = tileSize;
    renderConfig0.scaling = scaling;
    renderConfig0.overlap = overlap;
    renderConfig0.tta = true;

    trt::Img2Img engine;
    engine.setLogCallback(callback);
    engine.load(modelPath, renderConfig1);
    engine.load(modelPath, renderConfig0);

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
#endif
    return 0;
}