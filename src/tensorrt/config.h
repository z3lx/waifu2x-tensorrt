#ifndef WAIFU2X_TENSORRT_TRT_CONFIG_H
#define WAIFU2X_TENSORRT_TRT_CONFIG_H

#include "helper.h"

namespace trt {
    enum class Precision {
        TF32,
        FP16
    };

    struct BuildConfig {
        int deviceId = 0;
        Precision precision = Precision::FP16;

        int minBatchSize = 1;
        int optBatchSize = 1;
        int maxBatchSize = 4;

        int minChannels = 3;
        int optChannels = 3;
        int maxChannels = 3;

        int minWidth = 64;
        int optWidth = 256;
        int maxWidth = 640;

        int minHeight = 64;
        int optHeight = 256;
        int maxHeight = 640;
    };

    struct RenderConfig {
        bool tta = false;
        int deviceId = 0;
        Precision precision = Precision::FP16;
        int batchSize = 1;
        int channels = 3;
        int height = 256;
        int width = 256;
        int scaling = 4;
        cv::Point2d overlap = cv::Point2d(0.125f, 0.125f);
    };
}

#endif //WAIFU2X_TENSORRT_TRT_CONFIG_H