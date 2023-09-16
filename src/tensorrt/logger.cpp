#include "logger.h"

#include <utility>

trt::Logger::Logger() = default;
trt::Logger::~Logger() = default;

void trt::Logger::setLogCallback(LogCallback callback) {
    logCallback = std::move(callback);
}

#define LOG(severity, message) log(severity, message, __FILE__, __FUNCTION__, __LINE__)
void trt::Logger::log(trt::Severity severity, const std::string& message, const std::string& file, const std::string& function, int line) {
    if (logCallback) {
        logCallback(severity, message, file, function, line);
    }
}

void trt::Logger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept {
    switch (severity) {
        case Severity::kINTERNAL_ERROR:
            LOG(critical, msg);
            break;
        case Severity::kERROR:
            LOG(error, msg);
            break;
        case Severity::kWARNING:
            LOG(warn, msg);
            break;
        case Severity::kINFO:
            LOG(info, msg);
            break;
        case Severity::kVERBOSE:
            LOG(debug, msg);
            break;
    }
}