#include "engine.h"

trt::Engine::Engine(trt::Config config)
    : config(config)
{
}

trt::Engine::~Engine() {

}

bool trt::Engine::load(const std::string& onnxModelPath) {
    return true;
}

#include <iostream>
bool trt::Engine::build(const std::string& onnxModelPath) {
    // Create builder
    gLogger.log(nvinfer1::ILogger::Severity::kINFO, "Building engine...");
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder) {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to create builder.");
        return false;
    }

    // Create network
    auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to create network.");
        return false;
    }

    // Create parser
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser) {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to create parser.");
        return false;
    }

    // Parse ONNX model
    gLogger.log(nvinfer1::ILogger::Severity::kINFO, "Parsing ONNX model...");
    auto parsed = parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    if (!parsed) {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to parse ONNX model.");
        return false;
    }

    // Create builder builderConfig
    auto builderConfig = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!builderConfig) {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to create builder builderConfig.");
        return false;
    }

    //Create optimization profile
    auto nbInputs = network->getNbInputs();
    auto profile = builder->createOptimizationProfile();
    for (int32_t i = 0; i < nbInputs; ++i) {
        const auto input = network->getInput(i);
        const auto inputName = input->getName();
        const auto inputDims = input->getDimensions();
        int32_t channels = inputDims.d[1];

        profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN,
            nvinfer1::Dims4 { config.minBatchSize, channels, config.minHeight, config.minWidth });
        profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT,
            nvinfer1::Dims4 { config.optBatchSize, channels, config.optHeight, config.optWidth });
        profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX,
            nvinfer1::Dims4 { config.maxBatchSize, channels, config.maxHeight, config.maxWidth });
    }
    builderConfig->addOptimizationProfile(profile);

    // Set precision
    if (config.precision == Precision::FP16) {
        if (!builder->platformHasFastFp16()) {
            gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Platform does not support FP16.");
            return false;
        }
        builderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else if (config.precision == Precision::FP32) {
        if (!builder->platformHasTf32()) {
            gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Platform does not support FP32.");
            return false;
        }
        builderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    builderConfig->setProfileStream(stream);

    // Build engine
    gLogger.log(nvinfer1::ILogger::Severity::kINFO, "Building engine...");
    std::unique_ptr<nvinfer1::IHostMemory> engine {
        builder->buildSerializedNetwork(*network, *builderConfig)
    };
    if (!engine) {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to build engine.");
        return false;
    }

    // Save engine
    gLogger.log(nvinfer1::ILogger::Severity::kINFO, "Saving engine...");
    std::string modelPath = onnxModelPath;
    if (!serializeConfigToPath(modelPath)) {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to serialize config to path.");
        return false;
    }
    std::ofstream engineFile(modelPath, std::ios::binary);
    engineFile.write(reinterpret_cast<const char*>(engine->data()), engine->size());
    engineFile.close();

    // Destroy stream
    cudaStreamDestroy(stream);
    return true;
}

bool trt::Engine::serializeConfigToPath(std::string& onnxModelPath) {
    const auto filenameIndex = onnxModelPath.find_last_of('/') + 1;
    onnxModelPath = onnxModelPath.substr(filenameIndex, onnxModelPath.find_last_of('.') - filenameIndex);

    // Append device name
    std::vector<std::string> deviceNames;
    getDeviceNames(deviceNames);
    if (config.deviceIndex >= deviceNames.size()) {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Invalid device index.");
        return false;
    }
    auto deviceName = deviceNames[config.deviceIndex];
    deviceName.erase(std::remove_if(deviceName.begin(), deviceName.end(), ::isspace), deviceName.end());
    onnxModelPath += "." + deviceName;

    // Append precision
    switch (config.precision) {
    case Precision::FP16:
        onnxModelPath += ".FP16";
        break;
    case Precision::FP32:
        onnxModelPath += ".FP32";
        break;
    }

    // Append dynamic shapes
    onnxModelPath += "." + std::to_string(config.minBatchSize);
    onnxModelPath += "." + std::to_string(config.optBatchSize);
    onnxModelPath += "." + std::to_string(config.maxBatchSize);
    onnxModelPath += "." + std::to_string(config.minWidth);
    onnxModelPath += "." + std::to_string(config.optWidth);
    onnxModelPath += "." + std::to_string(config.maxWidth);
    onnxModelPath += "." + std::to_string(config.minHeight);
    onnxModelPath += "." + std::to_string(config.optHeight);
    onnxModelPath += "." + std::to_string(config.maxHeight);

    // Append extension
    onnxModelPath += ".trt";

    return true;
}

void trt::Engine::getDeviceNames(std::vector<std::string> &deviceNames) {
    int32_t deviceCount;
    cudaGetDeviceCount(&deviceCount);
    deviceNames.clear();
    deviceNames.reserve(deviceCount);
    for (int32_t i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp {};
        cudaGetDeviceProperties(&deviceProp, i);
        deviceNames.emplace_back(deviceProp.name);
    }
}
