#ifndef WAIFU2X_TENSORRT_TRT_LOGGER_H
#define WAIFU2X_TENSORRT_TRT_LOGGER_H

#include <NvInferRuntimeBase.h>
#include <functional>
#include <string>

#define LOG(severity, message) log(severity, message, __FUNCTION__, __LINE__)

namespace trt {
    enum Severity {
        critical,
        error,
        warn,
        info,
        debug,
        trace
    };

    using MessageCallback = std::function<void(trt::Severity, const std::string&)>;
    using ProgressCallback = std::function<void(int, int, double)>;

    class Logger : public nvinfer1::ILogger {
    public:
        Logger();
        ~Logger() override;

        void setMessageCallback(MessageCallback callback);
        void setProgressCallback(ProgressCallback callback);

        void log(trt::Severity severity, const std::string& message);
        void log(trt::Severity severity, const std::string& message, const std::string& function, int line);
        void log(ILogger::Severity severity, const char* msg) noexcept override;
        void log(int current, int total, double speed);

    private:
        MessageCallback messageCallback{};
        ProgressCallback progressCallback{};
    };
}

#endif //WAIFU2X_TENSORRT_TRT_LOGGER_H