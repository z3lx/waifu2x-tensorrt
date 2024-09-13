#include "img2img.h"

trt::Img2Img::Img2Img() {

}

trt::Img2Img::~Img2Img() {

}

// TODO: error handling and stream deallocation
// Set cuda device when building the engine
bool trt::Img2Img::build(const std::string& onnxModelPath, const BuilderConfig& config) {
    PLOG(plog::info) << "Img2Img build started with configuration"
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

// TODO: Check if configs are compatible
// TODO: error handling
bool trt::Img2Img::load(const std::string& modelPath, trt::InferrerConfig& config) {
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

    inferrerConfig = config;

    return true;
}

bool trt::Img2Img::infer(std::vector<cv::cuda::GpuMat>& inputs, std::vector<cv::cuda::GpuMat>& outputs) {
    // Check batch size
    if (inputs.size() != inferrerConfig.inputShape.d[0]) {
        PLOG(plog::error) << "Invalid batch size.";
        PLOG(plog::error) << "Expected " << inferrerConfig.inputShape.d[0] << ", got " << inputs.size() << ".";
        return false;
    }

    // Check image size
    for (const auto& mat : inputs) {
        //if (mat.channels() != inferrerConfig.inputShape.d[1]) {
        //    PLOG(plog::error) << "Invalid number of channels.";
        //    PLOG(plog::error) << "Expected " << inferrerConfig.inputShape.d[1] << ", got " << mat.channels() << ".";
        //    return false;
        //}

        if (mat.rows != inferrerConfig.inputShape.d[2]) {
            PLOG(plog::error) << "Invalid height.";
            PLOG(plog::error) << "Expected " << inferrerConfig.inputShape.d[2] << ", got " << mat.rows << ".";
            return false;
        }

        if (mat.cols != inferrerConfig.inputShape.d[3]) {
            PLOG(plog::error) << "Invalid width.";
            PLOG(plog::error) << "Expected " << inferrerConfig.inputShape.d[3] << ", got " << mat.cols << ".";
            return false;
        }
    }

    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    context->setOptimizationProfileAsync(0, stream);
    context->setInputShape(engine->getIOTensorName(0), inferrerConfig.inputShape);

    auto blob = blobFromImages(inputs, true);
    auto* ptr = blob.ptr<void>();
    auto size = blob.channels() * blob.rows * blob.cols * sizeof(float);
    cudaMemcpyAsync(buffers[0].first, ptr, size, cudaMemcpyDeviceToDevice, stream);

    // Set input and output addresses
    for (int i = 0; i < buffers.size(); ++i) {
        if (!context->setTensorAddress(engine->getIOTensorName(i), buffers[i].first)) {
            PLOG(plog::error) << "Failed to set tensor address for tensor \"" << engine->getIOTensorName(i) << "\".";
            return false;
        }
    }

    // Execute inference
    PLOG(plog::info) << "Executing inference...";
    if (!context->enqueueV3(stream)) {
        PLOG(plog::error) << "Failed to execute inference.";
        return false;
    }

    // Copy output to host
    // FIX: ONLY ONE BATCH
    std::vector<cv::cuda::GpuMat> temp(inferrerConfig.inputShape.d[0]);
    for (int i = 0; i < temp.size(); ++i) {
        auto shape = context->getTensorShape(engine->getIOTensorName(1));
        temp[i].create(shape.d[2], shape.d[3], CV_32FC3);
        cudaMemcpyAsync(temp[i].ptr<void>(), buffers[1].first, buffers[1].second, cudaMemcpyDeviceToDevice, stream);
    }

    outputs.clear();
    outputs = imagesFromBlob(temp[0], context->getTensorShape(engine->getIOTensorName(1)), true);

    // Destroy stream
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return true;
}

cv::cuda::GpuMat trt::Img2Img::blobFromImages(std::vector<cv::cuda::GpuMat>& images, bool normalize) {
    cv::cuda::GpuMat gpu_dst(1, images.size() * images[0].channels() * images[0].rows * images[0].cols, CV_8U);

    size_t width = images[0].cols * images[0].rows;
    for (size_t img = 0; img < images.size(); img++) {
        std::vector<cv::cuda::GpuMat> channels {
            cv::cuda::GpuMat(images[0].rows, images[0].cols, CV_8U, &(gpu_dst.ptr()[0 * width + 3 * width * img])),
            cv::cuda::GpuMat(images[0].rows, images[0].cols, CV_8U, &(gpu_dst.ptr()[1 * width + 3 * width * img])),
            cv::cuda::GpuMat(images[0].rows, images[0].cols, CV_8U, &(gpu_dst.ptr()[2 * width + 3 * width * img]))
        };
        cv::cuda::split(images[img], channels);  // HWC -> CHW
    }

    cv::cuda::GpuMat mfloat;
    if (normalize) {
        // [0.f, 1.f]
        gpu_dst.convertTo(mfloat, CV_32FC3, 1.f / 255.f);
    } else {
        // [0.f, 255.f]
        gpu_dst.convertTo(mfloat, CV_32FC3);
    }

    return mfloat;
}

std::vector<cv::cuda::GpuMat> trt::Img2Img::imagesFromBlob(cv::cuda::GpuMat& blob, nvinfer1::Dims32 shape, bool denormalize) {
    std::vector<cv::cuda::GpuMat> images(shape.d[0]);

    int width = shape.d[2] * shape.d[3] * sizeof(float);
    for (int i = 0; i < images.size(); ++i) {
        cv::cuda::GpuMat image;
        image.create(shape.d[2], shape.d[3], CV_32FC3);

        std::vector<cv::cuda::GpuMat> channels {
            cv::cuda::GpuMat(shape.d[2], shape.d[3], CV_32F, &(blob.ptr()[0 * width + 3 * width * i])),
            cv::cuda::GpuMat(shape.d[2], shape.d[3], CV_32F, &(blob.ptr()[1 * width + 3 * width * i])),
            cv::cuda::GpuMat(shape.d[2], shape.d[3], CV_32F, &(blob.ptr()[2 * width + 3 * width * i]))
        };

        cv::cuda::merge(channels, image);
        image.convertTo(images[0], CV_8UC3, denormalize ? 255.f : 1.f);
    }
    return images;
}

bool trt::Img2Img::serializeConfig(std::string& path, const BuilderConfig &config) {
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

bool trt::Img2Img::deserializeConfig(const std::string& path, BuilderConfig& config) {
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

void trt::Img2Img::getDeviceNames(std::vector<std::string> &deviceNames) {
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
