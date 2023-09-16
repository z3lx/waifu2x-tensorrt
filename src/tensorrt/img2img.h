#ifndef WAIFU2X_TENSORRT_TRT_IMG2IMG_H
#define WAIFU2X_TENSORRT_TRT_IMG2IMG_H

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <NvOnnxParser.h>
#include <NvInfer.h>

#include "config.h"
#include "helper.h"
#include "logger.h"
#include "utilities/time.h"

namespace trt {
    class Img2Img {
    public:
        Img2Img();
        virtual ~Img2Img();
        bool build(const std::string& path, const BuildConfig& config);
        bool load(const std::string& path, RenderConfig& config);
        bool render(cv::cuda::GpuMat& input, cv::cuda::GpuMat& output);
        void setLogCallback(LogCallback callback);

    private:
        Logger logger;

        cv::cuda::Stream stream{};
        std::vector<std::pair<void*, size_t>> buffers;
        RenderConfig renderConfig;
        cv::Size2i inputTileSize;
        cv::Size2i outputTileSize;
        cv::Size2i scaledInputTileSize;
        cv::Size2i scaledOutputTileSize;
        cv::Point2i inputOverlap;
        cv::Point2i scaledOutputOverlap;
        std::vector<cv::cuda::GpuMat> weights;

        std::unique_ptr<nvinfer1::IRuntime> runtime;
        std::unique_ptr<nvinfer1::ICudaEngine> engine;
        std::unique_ptr<nvinfer1::IExecutionContext> context;

        bool infer(const std::vector<cv::cuda::GpuMat>& inputs, std::vector<cv::cuda::GpuMat>& outputs);
        static cv::cuda::GpuMat blobFromImages(const std::vector<cv::cuda::GpuMat>& batch, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        static std::vector<cv::cuda::GpuMat> imagesFromBlob(void* blobPtr, nvinfer1::Dims32 shape, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        static cv::cuda::GpuMat padRoi(const cv::cuda::GpuMat& input, const cv::Rect2i& roi, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        static std::vector<cv::cuda::GpuMat> generateTileWeights(const cv::Point2i& overlap, const cv::Size2i& size, cv::cuda::Stream& stream = cv::cuda::Stream::Null());

        static bool serializeConfig(std::string& path, const BuildConfig& config);
        static bool deserializeConfig(const std::string& path, BuildConfig& config);
        static void getDeviceNames(std::vector<std::string>& deviceNames);
    };
}

#endif //WAIFU2X_TENSORRT_TRT_IMG2IMG_H