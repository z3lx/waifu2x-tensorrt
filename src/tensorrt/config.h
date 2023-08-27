#ifndef WAIFU2X_TENSORRT_TRT_CONFIG_H
#define WAIFU2X_TENSORRT_TRT_CONFIG_H

#include <cstdint>

namespace trt {
    enum class Precision {
        FP32,
        FP16
    };

    struct Config {
        Precision precision = Precision::FP16;
        int deviceIndex = 0;

        int32_t minBatchSize = 1;
        int32_t optBatchSize = 1;
        int32_t maxBatchSize = 4;

        int32_t minWidth = 64;
        int32_t optWidth = 256;
        int32_t maxWidth = 640;

        int32_t minHeight = 64;
        int32_t optHeight = 256;
        int32_t maxHeight = 640;

        bool operator==(const Config& other) const {
            return precision == other.precision &&
                deviceIndex == other.deviceIndex &&
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

        bool operator!=(const Config& other) const {
            return !(*this == other);
        }
    };
}

#endif //WAIFU2X_TENSORRT_TRT_CONFIG_H