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

    std::string imagePath = R"(C:\waifu2x-tensorrt\images\image.png)";
    std::string modelPath0 = R"(C:\waifu2x-tensorrt\models\cunet\art\noise3_scale2x.NVIDIAGeForceRTX3060Ti.FP16.1.1.1.256.256.256.256.256.256.trt)";
    //std::string modelPath1 = R"(C:\waifu2x-tensorrt\models\swin_unet\art\noise3_scale4x.NVIDIAGeForceRTX3060Ti.FP16.1.1.1.64.256.256.64.256.256.trt)";
    std::string modelPath1 = R"(C:\waifu2x-tensorrt\models\swin_unet\art\noise1_scale4x.NVIDIAGeForceRTX3060Ti.FP16.1.1.1.256.256.256.256.256.256.trt)";
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
    engine.load(modelPath0, inferConfig);
    engine.load(modelPath1, inferConfig);

    cv::Mat image = cv::imread(imagePath);
    cv::cuda::GpuMat input;
    input.upload(image);
    cv::cuda::cvtColor(input, input, cv::COLOR_BGR2RGB);
    std::vector<cv::cuda::GpuMat> inputs { input };
    std::vector<cv::cuda::GpuMat> outputs;

    cv::TickMeter tm;
    for (int i = 0; i < 10; ++i) {
        tm.start();
        engine.infer(inputs, outputs);
        tm.stop();
        PLOG(plog::info) << "Inference took " << tm.getTimeMilli() << " ms.";
        tm.reset();
    }

    cv::Mat output;
    outputs[0].download(output);
    cv::cvtColor(output, output, cv::COLOR_RGB2BGR);
    cv::imshow("output", output);
    cv::waitKey(-1);

    return 0;
}