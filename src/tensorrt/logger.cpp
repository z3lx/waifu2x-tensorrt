#include "logger.h"

void trt::Logger::log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept {
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
