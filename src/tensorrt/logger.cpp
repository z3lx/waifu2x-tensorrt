#include "logger.h"

void trt::Logger::log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept {
    std::cout << msg << std::endl;
}
