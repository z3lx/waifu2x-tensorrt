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
        int32_t inputC = inputDims.d[1];

        profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{ config.minBatchSize, inputC, config.minHeight, config.minWidth });
        profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{ config.optBatchSize, inputC, config.optHeight, config.optWidth });
        profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{ config.maxBatchSize, inputC, config.maxHeight, config.maxWidth });
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
    std::ofstream engineFile("engine.trt", std::ios::binary); //CHANGE NAME
    engineFile.write(reinterpret_cast<const char*>(engine->data()), engine->size());
    engineFile.close();

    // Destroy stream
    cudaStreamDestroy(stream);
    return true;
}
