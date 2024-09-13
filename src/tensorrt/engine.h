#ifndef WAIFU2X_TENSORRT_TRT_ENGINE_H
#define WAIFU2X_TENSORRT_TRT_ENGINE_H

#include <filesystem>
#include <fstream>
#include <memory>
#include <plog/Log.h>
#include <plog/Severity.h>
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
        bool load(const std::string& modelPath);
        bool build(const std::string& onnxModelPath);

    private:
        Logger gLogger;
        Config config;

        bool serializeConfig(std::string& onnxModelPath) const;
        static bool deserializeConfig(const std::string& trtEnginePath, Config &trtEngineConfig);
        static void getDeviceNames(std::vector<std::string>& deviceNames);
    };
}

#endif //WAIFU2X_TENSORRT_TRT_ENGINE_H