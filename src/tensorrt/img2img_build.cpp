#include "img2img.h"
#include "utilities/path.h"
#include "utilities/sha256.h"
#include <NvOnnxParser.h>
#include <fstream>

// TODO: ADD INPUT TENSOR SHAPE CONSTRAINTS
// TODO: SUPPORT MULTIPLE OPTIMIZATION PROFILES
bool trt::Img2Img::build(const std::string& onnxModelPath, const BuildConfig& config) try {
    // Set cuda device
    try {
        cudaAssert(cudaSetDevice(config.deviceId));
    }
    catch (const std::exception& e) {
        logger.LOG(error, "Failed to set cuda device to device id "
            + std::to_string(config.deviceId) + ": " + std::string(e.what()) + ".");
        return false;
    }

    // Create builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder) {
        logger.LOG(error, "Failed to create infer builder.");
        return false;
    }

    // Create network
    auto flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flags));
    if (!network) {
        logger.LOG(error, "Failed to create network.");
        return false;
    }

    // Create parser
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (!parser) {
        logger.LOG(error, "Failed to create parser.");
        return false;
    }

    // Parse ONNX model
    auto parsed = parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kVERBOSE));
    if (!parsed) {
        logger.LOG(error, "Failed to parse ONNX model.");
        return false;
    }

    // Create builder config
    auto builderConfig = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!builderConfig) {
        logger.LOG(error, "Failed to create builder config.");
        return false;
    }

    // Configure builder optimization profile
    auto nbInputs = network->getNbInputs();
    auto profile = builder->createOptimizationProfile();
    for (int i = 0; i < nbInputs; ++i) {
        const auto input = network->getInput(i);
        const auto inputName = input->getName();
        const auto inputDims = input->getDimensions();
        int channels = inputDims.d[1];

        auto min = nvinfer1::Dims4{config.minBatchSize, channels, config.minHeight, config.minWidth};
        auto opt = nvinfer1::Dims4{config.optBatchSize, channels, config.optHeight, config.optWidth};
        auto max = nvinfer1::Dims4{config.maxBatchSize, channels, config.maxHeight, config.maxWidth};
        profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, min);
        profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, opt);
        profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, max);
    }
    if (builderConfig->addOptimizationProfile(profile) != 0) {
        logger.LOG(error, "Failed to add optimization profile.");
        return false;
    }

    // Configure builder precision
    if (config.precision == Precision::FP16) {
        if (!builder->platformHasFastFp16()) {
            logger.LOG(error, "Failed to set precision: platform does not support FP16");
            return false;
        }
        builderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else if (config.precision == Precision::TF32) {
        if (!builder->platformHasTf32()) {
            logger.LOG(error, "Failed to set precision: platform does not support TF32");
            return false;
        }
        builderConfig->setFlag(nvinfer1::BuilderFlag::kTF32);
    }

    // Configure builder stream
    // Initialize stream before?
    builderConfig->setProfileStream(cudaGetCudaStream(stream));

    // Build network
    std::unique_ptr<nvinfer1::IHostMemory> serializedNetwork {
        builder->buildSerializedNetwork(*network, *builderConfig)
    };
    if (!serializedNetwork) {
        logger.LOG(error, "Failed to build serialized network.");
        return false;
    }

    // Serialize network
    const auto basePath = utils::removeFileExtension(onnxModelPath) + "_" + getConfigHash(config).substr(0, 16);
    const auto configPath = basePath + ".json";
    const auto enginePath = basePath + ".trt";
    serializeConfig(configPath, config);
    try {
        std::ofstream engineFile(enginePath, std::ios::binary);
        engineFile.write(
            reinterpret_cast<const char*>(serializedNetwork->data()),
            static_cast<long long>(serializedNetwork->size())
        );
    }
    catch (const std::exception& e) {
        logger.LOG(error, "Failed to serialize network to disk: " + std::string(e.what()) + ".");
        return false;
    }

    return true;
}
catch (const std::exception& e) {
    logger.LOG(error, "Engine build failed unexpectedly: " + std::string(e.what()) + ".");
    return false;
}

std::string trt::Img2Img::getConfigHash(const trt::BuildConfig& config) {
    std::ostringstream oss;
    auto deviceName = cudaGetDeviceName(config.deviceId);
    deviceName.erase(std::remove_if(deviceName.begin(), deviceName.end(), ::isspace), deviceName.end());
    oss << deviceName << ".";
    switch (config.precision) {
        case Precision::FP16:
            oss << "FP16";
            break;
        case Precision::TF32:
            oss << "TF32";
            break;
    }
    oss << ".";
    oss << config.minBatchSize << "." << config.optBatchSize << "." << config.maxBatchSize << "."
        << config.minChannels << "." << config.optChannels << "." << config.maxChannels << "."
        << config.minWidth << "." << config.optWidth << "." << config.maxWidth << "."
        << config.minHeight << "." << config.optHeight << "." << config.maxHeight;
    return utils::sha256(oss.str());
}