#include "engine.h"

trt::SuperResEngine::SuperResEngine() {

}

trt::SuperResEngine::~SuperResEngine() {

}

// TODO: Check if configs are compatible
// TODO: error handling
bool trt::SuperResEngine::load(const std::string& modelPath, trt::InferrerConfig config) {
    cudaSetDevice(config.deviceId);

    // Read engine to buffer
    PLOG(plog::info) << "Reading engine...";
    std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(fileSize);
    if (!file.read(buffer.data(), fileSize)) {
        PLOG(plog::error) << "Failed to read engine.";
        return false;
    }

    // Create runtime
    PLOG(plog::info) << "Creating runtime...";
    runtime.reset();
    runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    if (!runtime) {
        PLOG(plog::error) << "Failed to create runtime.";
        return false;
    }

    // Deserialize engine
    PLOG(plog::info) << "Deserializing engine...";
    engine.reset();
    engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!engine) {
        PLOG(plog::error) << "Failed to deserialize engine.";
        return false;
    }

    // Validate engine file
    if (engine->getNbIOTensors() != 2) {
        PLOG(plog::error) << "Invalid number of IO tensors.";
        PLOG(plog::error) << "Expected 2, got " << engine->getNbIOTensors() << ".";
        return false;
    }

    int nbDims = engine->getTensorShape(engine->getIOTensorName(0)).nbDims;
    if (nbDims != 4) {
        PLOG(plog::error) << "Invalid input tensor shape.";
        PLOG(plog::error) << "Expected 4, got " << nbDims << ".";
        return false;
    }

    nbDims = engine->getTensorShape(engine->getIOTensorName(1)).nbDims;
    if (nbDims != 4) {
        PLOG(plog::error) << "Invalid output tensor shape.";
        PLOG(plog::error) << "Expected 4, got " << nbDims << ".";
        return false;
    }

    // Create execution context
    PLOG(plog::info) << "Creating execution context...";
    context.reset();
    context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context) {
        PLOG(plog::error) << "Failed to create execution context.";
        return false;
    }

    // Set input shape
    if (!context->setInputShape(engine->getIOTensorName(0), config.inputShape)) {
        PLOG(plog::error) << "Failed to set shape for input tensor \"" << engine->getIOTensorName(0) << "\".";
        return false;
    }

    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocate buffers
    buffers.resize(2);

    // Allocate IO tensors
    PLOG(plog::info) << "Allocating memory for IO tensors...";
    {
        auto tensorShape = context->getTensorShape(engine->getIOTensorName(0));
        auto tensorSize = tensorShape.d[0] * tensorShape.d[1] * tensorShape.d[2] * tensorShape.d[3];
        buffers[0].first = nullptr;
        buffers[0].second = tensorSize * sizeof(float);
        cudaMallocAsync(&buffers[0].first, buffers[0].second, stream);
        input.clear();
        input.resize(tensorSize);

        tensorShape = context->getTensorShape(engine->getIOTensorName(1));
        tensorSize = tensorShape.d[0] * tensorShape.d[1] * tensorShape.d[2] * tensorShape.d[3];
        buffers[1].first = nullptr;
        buffers[1].second = tensorSize * sizeof(float);
        cudaMallocAsync(&buffers[1].first, buffers[1].second, stream);
        output.clear();
        output.resize(tensorSize);
    }

    // Destroy stream
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return true;
}

// TODO: error handling and stream deallocation
// Set cuda device when building the engine
bool trt::SuperResEngine::build(const std::string& onnxModelPath, const BuilderConfig& config) {
    PLOG(plog::info) << "SuperResEngine build started with configuration"
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
    auto parsed = parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kVERBOSE));
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
    if (!serializeConfig(modelPath, config)) {
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

bool trt::SuperResEngine::deserializeConfig(const std::string& path, BuilderConfig& config) {
    std::string trtModelName = path.substr(path.find_last_of('/') + 1);
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

    config.deviceIndex = deviceIndex;
    config.precision = tokens[2] == "FP16" ? Precision::FP16 : Precision::TF32;
    config.minBatchSize = std::stoi(tokens[3]);
    config.optBatchSize = std::stoi(tokens[4]);
    config.maxBatchSize = std::stoi(tokens[5]);
    config.minWidth = std::stoi(tokens[6]);
    config.optWidth = std::stoi(tokens[7]);
    config.maxWidth = std::stoi(tokens[8]);
    config.minHeight = std::stoi(tokens[9]);
    config.optHeight = std::stoi(tokens[10]);
    config.maxHeight = std::stoi(tokens[11]);

    return true;
}

bool trt::SuperResEngine::serializeConfig(std::string& path, const BuilderConfig &config) {
    const auto filenameIndex = path.find_last_of('/') + 1;
    path = path.substr(filenameIndex, path.find_last_of('.') - filenameIndex);

    // Append device name
    std::vector<std::string> deviceNames;
    getDeviceNames(deviceNames);
    if (config.deviceIndex >= deviceNames.size()) {
        PLOG(plog::error) << "Invalid device index.";
        return false;
    }
    auto deviceName = deviceNames[config.deviceIndex];
    deviceName.erase(std::remove_if(deviceName.begin(), deviceName.end(), ::isspace), deviceName.end());
    path += "." + deviceName;

    // Append precision
    switch (config.precision) {
    case Precision::FP16:
        path += ".FP16";
        break;
    case Precision::TF32:
        path += ".TF32";
        break;
    }

    // Append dynamic shapes
    path += "." + std::to_string(config.minBatchSize);
    path += "." + std::to_string(config.optBatchSize);
    path += "." + std::to_string(config.maxBatchSize);
    path += "." + std::to_string(config.minWidth);
    path += "." + std::to_string(config.optWidth);
    path += "." + std::to_string(config.maxWidth);
    path += "." + std::to_string(config.minHeight);
    path += "." + std::to_string(config.optHeight);
    path += "." + std::to_string(config.maxHeight);

    // Append extension
    path += ".trt";

    return true;
}

void trt::SuperResEngine::getDeviceNames(std::vector<std::string> &deviceNames) {
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
