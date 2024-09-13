#include "img2img.h"

trt::Img2Img::Img2Img() {

}

trt::Img2Img::~Img2Img() {
    for (auto& buffer : buffers) {
        cudaAssert(cudaFree(buffer.first));
    }
}

// TODO: ADD INPUT TENSOR SHAPE CONSTRAINTS
// TODO: SUPPORT MULTIPLE OPTIMIZATION PROFILES
// TODO: CHANGE MODEL CONFIGURATION SERIALIZATION
bool trt::Img2Img::build(const std::string& onnxModelPath, const BuilderConfig& config) try {
    // Set cuda device
    try {
        trt::cudaAssert(cudaSetDevice(config.deviceId));
    }
    catch (const std::exception& e) {
        throw std::runtime_error("could not set cuda device to device id "
            + std::to_string(config.deviceId) + " (" + e.what() + ")");
    }

    // Create builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder) {
        throw std::runtime_error("could not create builder");
    }

    // Create network
    auto flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flags));
    if (!network) {
        throw std::runtime_error("could note create network");
    }

    // Create parser
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser) {
        throw std::runtime_error("could not create parser");
    }

    // Parse ONNX model
    auto parsed = parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kVERBOSE));
    if (!parsed) {
        throw std::runtime_error("could not parse ONNX model");
    }

    // Create builder config
    auto builderConfig = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!builderConfig) {
        throw std::runtime_error("could not create builder config");
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
        throw std::runtime_error("could not add optimization profile");
    }

    // Configure builder precision
    if (config.precision == Precision::FP16) {
        if (!builder->platformHasFastFp16()) {
            throw std::runtime_error("could not set precision (platform does not support FP16)");
        }
        builderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else if (config.precision == Precision::TF32) {
        if (!builder->platformHasTf32()) {
            throw std::runtime_error("could not set precision (platform does not support TF32)");
        }
        builderConfig->setFlag(nvinfer1::BuilderFlag::kTF32);
    }

    // Create stream
    cudaStream_t stream;
    try {
        cudaAssert(cudaStreamCreate(&stream));
    }
    catch (const std::exception& e) {
        throw std::runtime_error("could not create cuda stream (" + std::string(e.what()) + ")");
    }

    // Configure builder stream
    builderConfig->setProfileStream(stream);

    // Build engine
    std::unique_ptr<nvinfer1::IHostMemory> engine {
        builder->buildSerializedNetwork(*network, *builderConfig)
    };
    if (!engine) {
        throw std::runtime_error("could not build serialized network");
    }

    // Serialize engine
    try {
        std::string modelPath = onnxModelPath;
        serializeConfig(modelPath, config);
        std::ofstream engineFile(modelPath, std::ios::binary);
        engineFile.write(reinterpret_cast<const char*>(engine->data()), engine->size());
        engineFile.close();
    }
    catch (const std::exception& e) {
        cudaAssert(cudaStreamDestroy(stream));
        throw std::runtime_error("could not serialize engine to disk (" + std::string(e.what()) + ")");
    }

    cudaAssert(cudaStreamDestroy(stream));
    return true;
}
catch (const std::exception& e) {
    PLOG(plog::error) << "Failed to build engine: " << e.what() << ".";
    return false;
}

// TODO: ENSURE CONFIGURATION COMPATIBILITY
// TODO: SET OPTIMIZATION PROFILE VIA nvinfer1::IExecutionContext::setOptimizationProfileAsync
bool trt::Img2Img::load(const std::string& modelPath, trt::InferrerConfig& config) try {
    // Set cuda device
    try {
        trt::cudaAssert(cudaSetDevice(config.deviceId));
    }
    catch (const std::exception& e) {
        throw std::runtime_error("could not set cuda device to device id "
            + std::to_string(config.deviceId) + " (" + e.what() + ")");
    }

    // Read engine
    std::vector<char> buffer;
    try {
        std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
        std::streamsize fileSize = file.tellg();
        buffer.resize(fileSize);
        file.seekg(0, std::ios::beg);
        file.read(buffer.data(), fileSize);
    }
    catch (const std::exception& e) {
        throw std::runtime_error("could not read engine file to buffer (" + std::string(e.what()) + ")");
    }

    // Destroy existing engine
    if (context || engine || runtime) {
        context.reset();
        engine.reset();
        //runtime.reset();
    }

    // Create runtime if necessary
    if (!runtime) {
        runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
        if (!runtime) {
            throw std::runtime_error("could not create infer runtime");
        }
    }

    // Deserialize engine
    engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!engine) {
        throw std::runtime_error("could not deserialize cuda engine from buffer");
    }

    // Validate engine
    const auto nbIOTensors = engine->getNbIOTensors();
    if (nbIOTensors != 2) {
        throw std::runtime_error("invalid number of IO tensors (expected 2, got " + std::to_string(nbIOTensors) + ")");
    }
    for (int i = 0; i < nbIOTensors; ++i) {
        const auto nbDims = engine->getTensorShape(engine->getIOTensorName(i)).nbDims;
        if (nbDims != 4) {
            throw std::runtime_error("invalid IO tensor shape (expected 4, got " + std::to_string(nbDims) + ")");
        }
    }

    // Create execution context
    context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context) {
        throw std::runtime_error("could not create execution context");
    }

    // Set tensor shapes
    nvinfer1::Dims4 inputShape{config.nbBatches, config.channels, config.height, config.width};
    if (!context->setInputShape(engine->getIOTensorName(0), inputShape)) {
        throw std::runtime_error("could not set shape for input tensor");
    }

    // Create stream
    cudaStream_t stream;
    try {
        cudaAssert(cudaStreamCreate(&stream));
    }
    catch (const std::exception& e) {
        throw std::runtime_error("could not create cuda stream (" + std::string(e.what()) + ")");
    }

    // Deallocate existing buffers
    if (!buffers.empty()) {
        try {
            for (auto& buffer : buffers) {
                cudaAssert(cudaFreeAsync(buffer.first, stream));
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("could not deallocate buffers (" + std::string(e.what()) + ")");
        }
        buffers.clear();
        buffers.reserve(nbIOTensors);
    }

    // Allocate buffers and set tensor addresses
    for (int i = 0; i < nbIOTensors; ++i) {
        auto tensorName = engine->getIOTensorName(i);
        auto tensorShape = context->getTensorShape(tensorName);
        auto tensorSize = tensorShape.d[0] * tensorShape.d[1] * tensorShape.d[2] * tensorShape.d[3];

        buffers.emplace_back(nullptr, tensorSize * sizeof(float));
        try {
            cudaAssert(cudaMallocAsync(&buffers[i].first, buffers[i].second, stream));
        }
        catch (const std::exception& e) {
            // Clean up
            for (auto& buffer : buffers) {
                cudaAssert(cudaFreeAsync(buffer.first, stream));
            }
            cudaAssert(cudaStreamDestroy(stream));

            throw std::runtime_error("could not allocate IO tensor buffers (" + std::string(e.what()) + ")");
        }

        if (!context->setTensorAddress(tensorName, buffers[i].first)) {
            throw std::runtime_error("could not set tensor address");
        }

        PLOG(plog::debug) << "Allocated " << (static_cast<double>(buffers[i].second) / pow(1024, 2))
            << " MiB on GPU for tensor \"" << tensorName << "\".";
    }

    inferrerConfig = config;
    cudaAssert(cudaStreamSynchronize(stream));
    cudaAssert(cudaStreamDestroy(stream));
    return true;
}
catch (const std::exception& e) {
    PLOG(plog::error) << "Failed to load engine: " << e.what() << ".";
    return false;
}

// TODO: VALIDATE THAT BATCHES WORK
bool trt::Img2Img::infer(std::vector<cv::cuda::GpuMat>& inputs, std::vector<cv::cuda::GpuMat>& outputs) try {
    // Check batch size
    if (inputs.size() != inferrerConfig.nbBatches) {
        throw std::runtime_error("invalid input batch size (expected "
            + std::to_string(inferrerConfig.nbBatches) + ", got " + std::to_string(inputs.size()) + ")");
    }

    // Check image size
    for (const auto& mat: inputs) {
        if (mat.channels() != inferrerConfig.channels) {
            throw std::runtime_error("invalid image channels (expected "
                + std::to_string(inferrerConfig.channels) + ", got " + std::to_string(mat.channels()) + ")");
        }

        if (mat.rows != inferrerConfig.height) {
            throw std::runtime_error("invalid image height (expected "
                + std::to_string(inferrerConfig.height) + ", got " + std::to_string(mat.rows) + ")");
        }

        if (mat.cols != inferrerConfig.width) {
            throw std::runtime_error("invalid image width (expected "
                + std::to_string(inferrerConfig.width) + ", got " + std::to_string(mat.cols) + ")");
        }
    }

    // Create stream
    cudaStream_t stream;
    try {
        cudaAssert(cudaStreamCreate(&stream));
    }
    catch (const std::exception& e) {
        throw std::runtime_error("could not create cuda stream (" + std::string(e.what()) + ")");
    }

    try {
        // Preprocess input
        auto blob = blobFromImages(inputs, true);

        // Copy blob to input tensor buffer
        cudaAssert(cudaMemcpyAsync(buffers[0].first, blob.ptr<void>(),
            buffers[0].second, cudaMemcpyDeviceToDevice, stream));

        // Enqueue inference
        if (!context->enqueueV3(stream)) {
            throw std::runtime_error("could not enqueue inference");
        }

        // Copy output to host
        std::vector<cv::cuda::GpuMat> temp(inferrerConfig.nbBatches);
        for (int i = 0; i < temp.size(); ++i) {
            auto shape = context->getTensorShape(engine->getIOTensorName(1));
            temp[i].create(shape.d[2], shape.d[3], CV_32FC3);
            cudaAssert(cudaMemcpyAsync(temp[i].ptr<void>(), buffers[1].first,
                buffers[1].second, cudaMemcpyDeviceToDevice, stream));
        }

        // Postprocess output
        outputs.clear();
        outputs = imagesFromBlob(temp[0], context->getTensorShape(engine->getIOTensorName(1)), true);
    }
    catch (const std::exception& e) {
        cudaAssert(cudaStreamDestroy(stream));
        throw e;
    }

    cudaAssert(cudaStreamSynchronize(stream));
    cudaAssert(cudaStreamDestroy(stream));
    return true;
}
catch (const std::exception& e) {
    PLOG(plog::error) << "Failed to infer: " << e.what() << ".";
    return false;
}

cv::cuda::GpuMat trt::Img2Img::blobFromImages(std::vector<cv::cuda::GpuMat>& images, bool normalize) {
    cv::cuda::GpuMat gpu_dst(1, images.size() * images[0].channels() * images[0].rows * images[0].cols, CV_8U);

    size_t width = images[0].cols * images[0].rows;
    for (size_t i = 0; i < images.size(); ++i) {
        std::vector<cv::cuda::GpuMat> channels {
            cv::cuda::GpuMat(images[0].rows, images[0].cols, CV_8U, &(gpu_dst.ptr()[0 * width + 3 * width * i])),
            cv::cuda::GpuMat(images[0].rows, images[0].cols, CV_8U, &(gpu_dst.ptr()[1 * width + 3 * width * i])),
            cv::cuda::GpuMat(images[0].rows, images[0].cols, CV_8U, &(gpu_dst.ptr()[2 * width + 3 * width * i]))
        };
        cv::cuda::split(images[i], channels);  // HWC -> CHW
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

bool trt::Img2Img::serializeConfig(std::string& path, const BuilderConfig& config) {
    const auto filenameIndex = path.find_last_of('/') + 1;
    path = path.substr(filenameIndex, path.find_last_of('.') - filenameIndex);

    // Append device name
    std::vector<std::string> deviceNames;
    getDeviceNames(deviceNames);
    if (config.deviceId >= deviceNames.size()) {
        PLOG(plog::error) << "Invalid device index.";
        return false;
    }
    auto deviceName = deviceNames[config.deviceId];
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

    config.deviceId = deviceIndex;
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

void trt::Img2Img::getDeviceNames(std::vector<std::string>& deviceNames) {
    int32_t deviceCount;
    cudaGetDeviceCount(&deviceCount);
    deviceNames.clear();
    deviceNames.reserve(deviceCount);
    for (int32_t i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp{};
        cudaGetDeviceProperties(&deviceProp, i);
        deviceNames.emplace_back(deviceProp.name);
    }
}
