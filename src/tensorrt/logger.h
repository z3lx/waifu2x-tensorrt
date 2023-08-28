#ifndef WAIFU2X_TENSORRT_TRT_LOGGER_H
#define WAIFU2X_TENSORRT_TRT_LOGGER_H

#include <iostream>
#include <NvInferRuntimeBase.h>
#include <plog/Log.h>
#include <plog/Severity.h>

namespace trt {
    class Logger : public nvinfer1::ILogger {
    public:
        Logger() = default;
        virtual ~Logger() = default;
        void log (Severity severity, const char* msg) noexcept override;
    };
}

#endif //WAIFU2X_TENSORRT_TRT_LOGGER_H
