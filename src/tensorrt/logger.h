#ifndef WAIFU2X_TENSORRT_TRT_LOGGER_H
#define WAIFU2X_TENSORRT_TRT_LOGGER_H

#include <NvInferRuntimeBase.h>
#include <functional>
#include <string>

#define LOG(severity, message) log(severity, message, __FILE__, __FUNCTION__, __LINE__)

namespace trt {
    enum Severity {
        critical,
        error,
        warn,
        info,
        debug,
        trace
    };

    using LogCallback = std::function<void(trt::Severity, const std::string&, const std::string&, const std::string&, int)>;

    class Logger : public nvinfer1::ILogger {
    public:
        Logger();
        ~Logger() override;

        void setLogCallback(LogCallback callback);
        void log(trt::Severity severity, const std::string& message, const std::string& file, const std::string& function, int line);
        void log(ILogger::Severity severity, const char* msg) noexcept override;

    private:
        LogCallback logCallback{};
    };
}

#endif //WAIFU2X_TENSORRT_TRT_LOGGER_H