#ifndef WAIFU2X_TENSORRT_TRT_ENGINE_H
#define WAIFU2X_TENSORRT_TRT_ENGINE_H

#include <fstream>
#include <memory>
#include <string>
#include <NvOnnxParser.h>
#include <NvInfer.h>

#include "config.h"
#include "logger.h"

namespace trt {
    class Engine {
    public:
        Engine(Config config);
        virtual ~Engine();
        bool load(const std::string& onnxModelPath);
        bool build(const std::string& onnxModelPath);

    private:
        Logger gLogger;
        Config config;
    };
}

#endif //WAIFU2X_TENSORRT_TRT_ENGINE_H