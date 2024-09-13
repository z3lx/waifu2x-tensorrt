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
        bool load(const std::string& path, RenderConfig& config);
        bool render(cv::cuda::GpuMat& input, cv::cuda::GpuMat& output);
        void setLogCallback(LogCallback callback);

    private:
        Logger logger;
        std::unique_ptr<nvinfer1::IRuntime> runtime;
        std::unique_ptr<nvinfer1::ICudaEngine> engine;
        std::unique_ptr<nvinfer1::IExecutionContext> context;

        // region Build
        static std::string getConfigHash(const BuildConfig& config);
        // endregion

        // region Load
        cv::cuda::Stream stream{};
        std::vector<std::pair<void*, size_t>> buffers;
        RenderConfig renderConfig;
        cv::Size2i inputTileSize;
        cv::Size2i outputTileSize;
        cv::Size2i scaledInputTileSize;
        cv::Size2i scaledOutputTileSize;
        cv::Point2i inputOverlap;
        cv::Point2i scaledOutputOverlap;
        std::array<cv::cuda::GpuMat, 4> weights;

        static std::string getEnginePath(const std::string& modelPath, const RenderConfig& config);
        static inline bool isCompatible(const RenderConfig& renderConfig, const BuildConfig& buildConfig);
        static inline bool isOptimized(const RenderConfig& renderConfig, const BuildConfig& buildConfig);
        static void createTileWeights(std::array<cv::cuda::GpuMat, 4>& weights, const cv::Point2i& overlap, const cv::Size2i& size, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        // endregion

        // region Infer
        bool infer(const std::vector<cv::cuda::GpuMat>& inputs, std::vector<cv::cuda::GpuMat>& outputs);
        static cv::cuda::GpuMat blobFromImages(const std::vector<cv::cuda::GpuMat>& batch, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        static std::vector<cv::cuda::GpuMat> imagesFromBlob(void* blobPtr, nvinfer1::Dims32 shape, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        // endregion

        // region Render
        std::vector<cv::cuda::GpuMat> ttaInputTiles;
        cv::cuda::GpuMat ttaOutputTile;
        cv::cuda::GpuMat tmpInputMat;
        cv::cuda::GpuMat tmpOutputMat;

        inline std::tuple<const int, std::vector<cv::Rect2i>, std::vector<cv::Rect2i>> calculateTiles(const cv::Rect2i& inputRect, const cv::Rect2i& outputRect);
        static cv::cuda::GpuMat padRoi(const cv::cuda::GpuMat& input, const cv::Rect2i& roi, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        inline void applyBlending(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, const cv::Rect2i& srcRect, const cv::Rect2i& dstRect);
        inline void applyAugmentation(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, const cv::Size2i& dstSize, int augmentationIndex);
        inline void reverseAugmentation(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, const cv::Size2i& dstSize, int augmentationIndex);
        // endregion

        // region Serialize
        static bool serializeConfig(const std::string& path, const BuildConfig& config);
        static bool deserializeConfig(const std::string& path, BuildConfig& config);
        // endregion
    };
}

#endif //WAIFU2X_TENSORRT_TRT_IMG2IMG_H