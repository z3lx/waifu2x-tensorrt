#include "logger.h"

trt::Logger::Logger() = default;
trt::Logger::~Logger() = default;

void trt::Logger::setMessageCallback(MessageCallback callback) {
    messageCallback = std::move(callback);
}

void trt::Logger::setProgressCallback(trt::ProgressCallback callback) {
    progressCallback = std::move(callback);
}

void trt::Logger::log(trt::Severity severity, const std::string& message) {
    if (messageCallback)
        messageCallback(severity, message);
}

void trt::Logger::log(trt::Severity severity, const std::string& message, const std::string& function, int line) {
    if (messageCallback)
        messageCallback(severity, "[" + function + "@" + std::to_string(line) + "] " + message);
}

void trt::Logger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept {
    switch (severity) {
        case Severity::kINTERNAL_ERROR:
            log(critical, msg);
            break;
        case Severity::kERROR:
            log(error, msg);
            break;
        case Severity::kWARNING:
            log(warn, msg);
            break;
        case Severity::kINFO:
            log(info, msg);
            break;
        case Severity::kVERBOSE:
            log(debug, msg);
            break;
    }
}

void trt::Logger::log(int current, int total, double speed) {
    if (progressCallback)
        progressCallback(current, total, speed);
}