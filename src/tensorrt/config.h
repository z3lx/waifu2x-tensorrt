#ifndef WAIFU2X_TENSORRT_TRT_CONFIG_H
#define WAIFU2X_TENSORRT_TRT_CONFIG_H

namespace trt {
    enum class Precision {
        TF32,
        FP16
    };

    struct BuildConfig {
        int deviceId = 0;
        Precision precision = Precision::FP16;

        int32_t minBatchSize = 1;
        int32_t optBatchSize = 1;
        int32_t maxBatchSize = 4;

        int32_t minWidth = 64;
        int32_t optWidth = 256;
        int32_t maxWidth = 640;

        int32_t minHeight = 64;
        int32_t optHeight = 256;
        int32_t maxHeight = 640;

        bool operator==(const BuildConfig& other) const {
            return precision == other.precision &&
                deviceId == other.deviceId &&
                minBatchSize == other.minBatchSize &&
                optBatchSize == other.optBatchSize &&
                maxBatchSize == other.maxBatchSize &&
                minWidth == other.minWidth &&
                optWidth == other.optWidth &&
                maxWidth == other.maxWidth &&
                minHeight == other.minHeight &&
                optHeight == other.optHeight &&
                maxHeight == other.maxHeight;
        }

        bool operator!=(const BuildConfig& other) const {
            return !(*this == other);
        }
    };

    struct RenderConfig {
        bool tta = false;
        int deviceId = 0;
        Precision precision = Precision::FP16;
        int nbBatches = 1;
        int channels = 3;
        int height = 256;
        int width = 256;
        cv::Point2i scaling = cv::Point2i(4, 4);
        cv::Point2d overlap = cv::Point2d(0.125f, 0.125f);
    };
}

#endif //WAIFU2X_TENSORRT_TRT_CONFIG_H