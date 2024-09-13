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
#include <plog/Log.h>
#include <plog/Severity.h>
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
        bool build(const std::string& path, const BuilderConfig& config);
        bool load(const std::string& path, InferrerConfig& config);
        bool infer(std::vector<cv::cuda::GpuMat>& inputs, std::vector<cv::cuda::GpuMat>& outputs);

    private:
        Logger gLogger;

        InferrerConfig inferrerConfig;
        std::vector<std::pair<void*, size_t>> buffers;

        std::unique_ptr<nvinfer1::IRuntime> runtime;
        std::unique_ptr<nvinfer1::ICudaEngine> engine;
        std::unique_ptr<nvinfer1::IExecutionContext> context;

        static cv::cuda::GpuMat blobFromImages(std::vector<cv::cuda::GpuMat>& batch, bool normalize);
        static std::vector<cv::cuda::GpuMat> imagesFromBlob(cv::cuda::GpuMat& blob, nvinfer1::Dims32 shape, bool denormalize);
        static bool serializeConfig(std::string& path, const BuilderConfig& config);
        static bool deserializeConfig(const std::string& path, BuilderConfig& config);
        static void getDeviceNames(std::vector<std::string>& deviceNames);
    };
}

#endif //WAIFU2X_TENSORRT_TRT_IMG2IMG_H