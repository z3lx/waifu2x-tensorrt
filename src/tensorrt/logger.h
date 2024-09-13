#ifndef WAIFU2X_TENSORRT_TRT_LOGGER_H
#define WAIFU2X_TENSORRT_TRT_LOGGER_H

#include <functional>
#include <string>
#include <NvInferRuntimeBase.h>

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
        void log(Severity severity, const char* msg) noexcept override;

    private:
        LogCallback logCallback{};
    };
}

#endif //WAIFU2X_TENSORRT_TRT_LOGGER_H