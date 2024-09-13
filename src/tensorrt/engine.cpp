#include "engine.h"

trt::Engine::Engine(trt::Config config)
    : config(config)
{
}

trt::Engine::~Engine() {

}

bool trt::Engine::load(const std::string& modelPath) {
    return true;
}

bool trt::Engine::build(const std::string& onnxModelPath) {
    PLOG(plog::info) << "Engine build started with configuration"
        << ": device = " << config.deviceIndex
        << ", precision = " << (config.precision == Precision::FP16 ? "FP16" : "TF32")
        << ", minBatchSize = " << config.minBatchSize
        << ", optBatchSize = " << config.optBatchSize
        << ", maxBatchSize = " << config.maxBatchSize
        << ", minWidth = " << config.minWidth
        << ", optWidth = " << config.optWidth
        << ", maxWidth = " << config.maxWidth
        << ", minHeight = " << config.minHeight
        << ", optHeight = " << config.optHeight
        << ", maxHeight = " << config.maxHeight
        << ".";

    // Create builder
    PLOG(plog::info) << "Creating builder...";
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder) {
        PLOG(plog::error) << "Failed to create builder.";
        return false;
    }

    // Create network
    PLOG(plog::info) << "Creating network...";
    auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        PLOG(plog::error) << "Failed to create network.";
        return false;
    }

    // Create parser
    PLOG(plog::info) << "Creating parser...";
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser) {
        PLOG(plog::error) << "Failed to create parser.";
        return false;
    }

    // Parse ONNX model
    PLOG(plog::info) << "Parsing ONNX model...";
    auto parsed = parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    if (!parsed) {
        PLOG(plog::error) << "Failed to parse ONNX model.";
        return false;
    }

    // Create builder config
    PLOG(plog::info) << "Creating builder config...";
    auto builderConfig = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!builderConfig) {
        PLOG(plog::error) << "Failed to create builder config.";
        return false;
    }

    //Create optimization profile
    PLOG(plog::info) << "Creating optimization profile...";
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
    PLOG(plog::info) << "Setting precision...";
    if (config.precision == Precision::FP16) {
        if (!builder->platformHasFastFp16()) {
            PLOG(plog::error) << "Platform does not support FP16.";
            return false;
        }
        builderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else if (config.precision == Precision::TF32) {
        if (!builder->platformHasTf32()) {
            PLOG(plog::error) << "Platform does not support TF32.";
            return false;
        }
        builderConfig->setFlag(nvinfer1::BuilderFlag::kTF32);
    }

    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    builderConfig->setProfileStream(stream);

    // Build engine
    PLOG(plog::info) << "Building engine...";
    std::unique_ptr<nvinfer1::IHostMemory> engine {
        builder->buildSerializedNetwork(*network, *builderConfig)
    };
    if (!engine) {
        PLOG(plog::error) << "Failed to build engine.";
        return false;
    }

    // Save engine
    PLOG(plog::info) << "Saving engine...";
    std::string modelPath = onnxModelPath;
    if (!serializeConfig(modelPath)) {
        PLOG(plog::error) << "Failed to serialize config to path.";
        return false;
    }
    std::ofstream engineFile(modelPath, std::ios::binary);
    engineFile.write(reinterpret_cast<const char*>(engine->data()), engine->size());
    engineFile.close();

    // Destroy stream
    cudaStreamDestroy(stream);
    return true;
}

bool trt::Engine::deserializeConfig(const std::string& trtEnginePath, Config &trtEngineConfig) {
    std::string trtModelName = trtEnginePath.substr(trtEnginePath.find_last_of('/') + 1);
    std::istringstream iss(trtModelName);
    std::string token;

    std::vector<std::string> tokens;
    while (std::getline(iss, token, '.')) {
        tokens.push_back(token);
    }

    if (tokens.size() != 13) {
        PLOG(plog::error) << "Invalid engine.";
        return false;
    }

    auto deviceIndex = -1;
    std::vector<std::string> deviceNames;
    getDeviceNames(deviceNames);
    for (int i = 0; i < deviceNames.size(); ++i) {
        auto deviceName = deviceNames[i];
        deviceName.erase(std::remove_if(deviceName.begin(), deviceName.end(), ::isspace), deviceName.end());
        if (deviceName == tokens[1]) {
            deviceIndex = i;
            break;
        }
    }

    if (deviceIndex == -1) {
        PLOG(plog::error) << "Invalid device.";
        return false;
    }

    trtEngineConfig.deviceIndex = deviceIndex;
    trtEngineConfig.precision = tokens[2] == "FP16" ? Precision::FP16 : Precision::TF32;
    trtEngineConfig.minBatchSize = std::stoi(tokens[3]);
    trtEngineConfig.optBatchSize = std::stoi(tokens[4]);
    trtEngineConfig.maxBatchSize = std::stoi(tokens[5]);
    trtEngineConfig.minWidth = std::stoi(tokens[6]);
    trtEngineConfig.optWidth = std::stoi(tokens[7]);
    trtEngineConfig.maxWidth = std::stoi(tokens[8]);
    trtEngineConfig.minHeight = std::stoi(tokens[9]);
    trtEngineConfig.optHeight = std::stoi(tokens[10]);
    trtEngineConfig.maxHeight = std::stoi(tokens[11]);

    return true;
}

bool trt::Engine::serializeConfig(std::string& onnxModelPath) const {
    const auto filenameIndex = onnxModelPath.find_last_of('/') + 1;
    onnxModelPath = onnxModelPath.substr(filenameIndex, onnxModelPath.find_last_of('.') - filenameIndex);

    // Append device name
    std::vector<std::string> deviceNames;
    getDeviceNames(deviceNames);
    if (config.deviceIndex >= deviceNames.size()) {
        PLOG(plog::error) << "Invalid device index.";
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
    case Precision::TF32:
        onnxModelPath += ".TF32";
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
