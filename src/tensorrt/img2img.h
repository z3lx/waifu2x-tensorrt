#ifndef WAIFU2X_TENSORRT_TRT_IMG2IMG_H
#define WAIFU2X_TENSORRT_TRT_IMG2IMG_H

#include "config.h"
#include "logger.h"
#include <NvInfer.h>
#include <opencv2/core/cuda.hpp>
#include <memory>
#include <queue>
#include <string>
#include <vector>

namespace trt {
    class Img2Img {
    public:
        Img2Img();
        virtual ~Img2Img();
        bool build(const std::string& path, const BuildConfig& config);
        bool load(const std::string& path, const RenderConfig& config);
        bool render(const cv::Mat& src, cv::Mat& dst);
        void setLogCallback(LogCallback callback);
        // setProgressCallback

    private:
        bool infer(const std::vector<cv::cuda::GpuMat>& inputs, std::vector<cv::cuda::GpuMat>& outputs);

        // Engine
        Logger logger;
        std::unique_ptr<nvinfer1::IRuntime> runtime;
        std::unique_ptr<nvinfer1::ICudaEngine> engine;
        std::unique_ptr<nvinfer1::IExecutionContext> context;

        // Inference
        cv::cuda::Stream stream;
        std::vector<std::pair<void*, size_t>> buffers;
        RenderConfig renderConfig;
        cv::cuda::GpuMat input;
        cv::cuda::GpuMat output;
        nvinfer1::Dims inputTensorShape;
        nvinfer1::Dims outputTensorShape;

        // Blending
        std::array<cv::cuda::GpuMat, 4> weights;

        // Augmentation
        std::vector<cv::cuda::GpuMat> ttaInputTiles;
        cv::cuda::GpuMat ttaOutputTile;
        cv::cuda::GpuMat tmpInputMat;
        cv::cuda::GpuMat tmpOutputMat;
    };
}

#endif //WAIFU2X_TENSORRT_TRT_IMG2IMG_H