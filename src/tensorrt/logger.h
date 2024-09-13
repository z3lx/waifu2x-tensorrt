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
        void log (Severity severity, const char* msg) noexcept override {
            switch (severity) {
                case Severity::kINTERNAL_ERROR:
                    PLOG(plog::fatal) << msg;
                    break;
                case Severity::kERROR:
                    PLOG(plog::error) << msg;
                    break;
                case Severity::kWARNING:
                    PLOG(plog::warning) << msg;
                    break;
                case Severity::kINFO:
                    PLOG(plog::info) << msg;
                    break;
                case Severity::kVERBOSE:
                    PLOG(plog::verbose) << msg;
                    break;
            }
        }
    };
}

#endif //WAIFU2X_TENSORRT_TRT_LOGGER_H
