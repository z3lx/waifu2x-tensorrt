#include "img2img.h"

trt::Img2Img::Img2Img() = default;

trt::Img2Img::~Img2Img() {
    for (auto& buffer : buffers) {
        cudaAssert(cudaFree(buffer.first));
    }
}

// TODO: ADD INPUT TENSOR SHAPE CONSTRAINTS
// TODO: SUPPORT MULTIPLE OPTIMIZATION PROFILES
// TODO: CHANGE MODEL CONFIGURATION SERIALIZATION
bool trt::Img2Img::build(const std::string& onnxModelPath, const BuildConfig& config) try {
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

    // Build network
    std::unique_ptr<nvinfer1::IHostMemory> serializedNetwork {
        builder->buildSerializedNetwork(*network, *builderConfig)
    };
    if (!serializedNetwork) {
        throw std::runtime_error("could not build serialized network");
    }

    // Serialize network
    try {
        std::string modelPath = onnxModelPath;
        serializeConfig(modelPath, config);
        std::ofstream engineFile(modelPath, std::ios::binary);
        engineFile.write(reinterpret_cast<const char*>(serializedNetwork->data()), static_cast<long long>(serializedNetwork->size()));
        engineFile.close();
    }
    catch (const std::exception& e) {
        cudaAssert(cudaStreamDestroy(stream));
        throw std::runtime_error("could not serialize network to disk (" + std::string(e.what()) + ")");
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
bool trt::Img2Img::load(const std::string& modelPath, trt::RenderConfig& config) try {
    // Set cuda device
    try {
        trt::cudaAssert(cudaSetDevice(config.deviceId));
    }
    catch (const std::exception& e) {
        throw std::runtime_error("could not set cuda device to device id "
            + std::to_string(config.deviceId) + " (" + e.what() + ")");
    }

    // Read engine
    std::vector<char> engineBuffer;
    try {
        std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
        std::streamsize fileSize = file.tellg();
        engineBuffer.resize(fileSize);
        file.seekg(0, std::ios::beg);
        file.read(engineBuffer.data(), fileSize);
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
    engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engineBuffer.data(), engineBuffer.size()));
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
    stream = cv::cuda::Stream(cudaStreamNonBlocking);

    // Deallocate existing buffers
    if (!buffers.empty()) {
        try {
            for (auto& buffer : buffers) {
                cudaAssert(cudaFreeAsync(buffer.first, static_cast<cudaStream_t>(stream.cudaPtr())));
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
            cudaAssert(cudaMallocAsync(&buffers[i].first, buffers[i].second, static_cast<cudaStream_t>(stream.cudaPtr())));
        }
        catch (const std::exception& e) {
            // Clean up
            for (auto& buffer : buffers) {
                cudaAssert(cudaFreeAsync(buffer.first, static_cast<cudaStream_t>(stream.cudaPtr())));
            }
            //cudaAssert(cudaStreamDestroy(static_cast<cudaStream_t>(stream.cudaPtr())));

            throw std::runtime_error("could not allocate IO tensor buffers (" + std::string(e.what()) + ")");
        }

        if (!context->setTensorAddress(tensorName, buffers[i].first)) {
            throw std::runtime_error("could not set tensor address");
        }

        PLOG(plog::debug) << "Allocated " << (static_cast<double>(buffers[i].second) / pow(1024, 2))
            << " MiB on GPU for tensor \"" << tensorName << "\".";
    }

    // TODO: ADD WARNING FOR ROUNDING ERRORS
    renderConfig = config;

    inputTileSize = {
        context->getTensorShape(engine->getIOTensorName(0)).d[3],
        context->getTensorShape(engine->getIOTensorName(0)).d[2]
    };

    outputTileSize = {
        context->getTensorShape(engine->getIOTensorName(1)).d[3],
        context->getTensorShape(engine->getIOTensorName(1)).d[2]
    };

    scaledOutputTileSize = {
        inputTileSize.width * renderConfig.scaling.x,
        inputTileSize.width * renderConfig.scaling.y
    };

    scaledInputTileSize = {
        static_cast<int>(std::lround(static_cast<double>(outputTileSize.width) / scaledOutputTileSize.width * inputTileSize.width)),
        static_cast<int>(std::lround(static_cast<double>(outputTileSize.height) / scaledOutputTileSize.height * inputTileSize.height))
    };

    inputOverlap = {
        static_cast<int>(std::lround(inputTileSize.width * renderConfig.overlap.x)),
        static_cast<int>(std::lround(inputTileSize.height * renderConfig.overlap.y))
    };

    scaledOutputOverlap = {
        static_cast<int>(std::lround(scaledOutputTileSize.width * renderConfig.overlap.x)),
        static_cast<int>(std::lround(scaledOutputTileSize.height * renderConfig.overlap.y))
    };

    weights = generateTileWeights(scaledOutputOverlap, outputTileSize, stream);

    return true;
}
catch (const std::exception& e) {
    PLOG(plog::error) << "Failed to load engine: " << e.what() << ".";
    return false;
}

bool trt::Img2Img::render(cv::cuda::GpuMat& input, cv::cuda::GpuMat& output) try {
    output.create(input.rows * renderConfig.scaling.x, input.cols * renderConfig.scaling.y, CV_32FC3);
    output.setTo(cv::Scalar(0, 0, 0));

    // Calculate tiling
    const cv::Point2i tiling = {
        static_cast<int>(std::lround(std::ceil(static_cast<double>(input.cols - inputOverlap.x) / (scaledInputTileSize.width - inputOverlap.x)))),
        static_cast<int>(std::lround(std::ceil(static_cast<double>(input.rows - inputOverlap.y) / (scaledInputTileSize.height - inputOverlap.y))))
    };
    const int tileCount = tiling.x * tiling.y;

    // Calculate tile rects
    std::vector<cv::Rect2i> inputTileRects;
    std::vector<cv::Rect2i> outputTileRects;
    inputTileRects.reserve(tiling.x * tiling.y);
    outputTileRects.reserve(tiling.x * tiling.y);
    for (int i = 0; i < tiling.y; ++i) {
        for (int j = 0; j < tiling.x; ++j) {
            // offset_border + offset_scaled_tile - offset_overlap
            cv::Rect2i inputTileRect;
            inputTileRect.x = -((inputTileSize.width - scaledInputTileSize.width) / 2) + (j * scaledInputTileSize.width) - (j * inputOverlap.x);
            inputTileRect.y = -((inputTileSize.height - scaledInputTileSize.height) / 2) + (i * scaledInputTileSize.height) - (i * inputOverlap.y);
            inputTileRect.width = inputTileSize.width;
            inputTileRect.height = inputTileSize.height;
            inputTileRects.emplace_back(inputTileRect);

            // offset_tile - offset_overlap
            cv::Rect2i outputTileRect;
            outputTileRect.x = j * outputTileSize.width - (j * scaledOutputOverlap.x);
            outputTileRect.y = i * outputTileSize.height - (i * scaledOutputOverlap.y);
            outputTileRect.width = outputTileRect.x + outputTileSize.width > output.cols ? output.cols - outputTileRect.x : outputTileSize.width;
            outputTileRect.height = outputTileRect.y + outputTileSize.height > output.rows ? output.rows - outputTileRect.y : outputTileSize.height;
            outputTileRects.emplace_back(outputTileRect);
        }
    }

    for (size_t i = 0; i < tileCount; i += renderConfig.nbBatches) {
        std::vector<cv::cuda::GpuMat> inputTiles;
        inputTiles.reserve(renderConfig.nbBatches);
        for (size_t j = 0; j < renderConfig.nbBatches; ++j) {
            if (i + j < tileCount) {
                inputTiles.emplace_back(padRoi(input, inputTileRects[i + j], stream));
            } else {
                inputTiles.emplace_back(inputTileSize.height, inputTileSize.width, CV_32FC3, cv::Scalar(0, 0, 0));
            }
        }

        std::vector<cv::cuda::GpuMat> outputTiles;
        outputTiles.reserve(renderConfig.nbBatches);
        infer(inputTiles, outputTiles);

        const bool overlapping = renderConfig.overlap.x != 0 || renderConfig.overlap.y != 0;
        for (size_t j = 0; j < renderConfig.nbBatches; ++j) {
            if (i + j == tileCount) {
                break;
            }
            cv::Rect2i& outputTileRect = outputTileRects[i + j];

            if (overlapping) {
                // Blend left
                if (outputTileRect.x != 0) {
                    cv::cuda::multiply(outputTiles[j], weights[3], outputTiles[j], 1, -1, stream);
                }

                // Blend top
                if (outputTileRect.y != 0) {
                    cv::cuda::multiply(outputTiles[j], weights[0], outputTiles[j], 1, -1, stream);
                }

                // Blend right
                if (outputTileRect.x + outputTileRect.width < output.cols) {
                    cv::cuda::multiply(outputTiles[j], weights[1], outputTiles[j], 1, -1, stream);
                }

                // Blend bottom
                if (outputTileRect.y + outputTileRect.height < output.rows) {
                    cv::cuda::multiply(outputTiles[j], weights[2], outputTiles[j], 1, -1, stream);
                }
            }

            cv::Rect2i roi(0, 0, outputTileRect.width, outputTileRect.height);
            cv::cuda::add(outputTiles[j](roi), output(outputTileRect), output(outputTileRect), cv::noArray(), -1, stream);
        }
    }
    output.convertTo(output, CV_8UC3, 255.0, stream);
    stream.waitForCompletion();

    return true;
}
catch (const std::exception& e) {
    PLOG(plog::error) << "Failed to render: " << e.what() << ".";
    return false;
}

bool trt::Img2Img::infer(const std::vector<cv::cuda::GpuMat>& inputs, std::vector<cv::cuda::GpuMat>& outputs) try {
    // Check batch size
    if (inputs.size() != renderConfig.nbBatches) {
        throw std::runtime_error("invalid input batch size (expected "
            + std::to_string(renderConfig.nbBatches) + ", got " + std::to_string(inputs.size()) + ")");
    }

    // Check image size
    for (const auto& mat: inputs) {
        if (mat.channels() != renderConfig.channels) {
            throw std::runtime_error("invalid image channels (expected "
                + std::to_string(renderConfig.channels) + ", got " + std::to_string(mat.channels()) + ")");
        }

        if (mat.rows != renderConfig.height) {
            throw std::runtime_error("invalid image height (expected "
                + std::to_string(renderConfig.height) + ", got " + std::to_string(mat.rows) + ")");
        }

        if (mat.cols != renderConfig.width) {
            throw std::runtime_error("invalid image width (expected "
                + std::to_string(renderConfig.width) + ", got " + std::to_string(mat.cols) + ")");
        }
    }

    try {
        // Preprocess input
        auto blob = blobFromImages(inputs, stream);

        // Copy blob to input tensor buffer
        cudaAssert(cudaMemcpyAsync(buffers[0].first, blob.ptr<void>(),
            buffers[0].second, cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream.cudaPtr())));

        // Enqueue inference
        if (!context->enqueueV3(static_cast<cudaStream_t>(stream.cudaPtr()))) {
            throw std::runtime_error("could not enqueue inference");
        }

        // Postprocess output
        outputs = imagesFromBlob(buffers[1].first, context->getTensorShape(engine->getIOTensorName(1)), stream);
    }
    catch (const std::exception& e) {
        //cudaAssert(cudaStreamDestroy(static_cast<cudaStream_t>(stream.cudaPtr())));
        throw e;
    }

    //cudaAssert(cudaStreamSynchronize(static_cast<cudaStream_t>(stream.cudaPtr())));
    //cudaAssert(cudaStreamDestroy(static_cast<cudaStream_t>(stream.cudaPtr())));
    return true;
}
catch (const std::exception& e) {
    PLOG(plog::error) << "Failed to infer: " << e.what() << ".";
    return false;
}

cv::cuda::GpuMat trt::Img2Img::blobFromImages(const std::vector<cv::cuda::GpuMat>& images, cv::cuda::Stream& stream) {
    cv::cuda::GpuMat blob(static_cast<int>(images.size()), images[0].channels() * images[0].rows * images[0].cols, CV_8U);

    size_t width = images[0].cols * images[0].rows;
    for (size_t i = 0; i < images.size(); ++i) {
        std::vector<cv::cuda::GpuMat> channels {
            cv::cuda::GpuMat(images[0].rows, images[0].cols, CV_8U, &(blob.ptr()[0 * width + 3 * width * i])),
            cv::cuda::GpuMat(images[0].rows, images[0].cols, CV_8U, &(blob.ptr()[1 * width + 3 * width * i])),
            cv::cuda::GpuMat(images[0].rows, images[0].cols, CV_8U, &(blob.ptr()[2 * width + 3 * width * i]))
        };
        cv::cuda::split(images[i], channels, stream);
    }

    blob.convertTo(blob, CV_32FC3, 1.0 / 255.0, stream);
    return blob;
}

std::vector<cv::cuda::GpuMat> trt::Img2Img::imagesFromBlob(void* blobPtr, nvinfer1::Dims32 shape, cv::cuda::Stream& stream) {
    std::vector<cv::cuda::GpuMat> images(shape.d[0]);

    size_t width = shape.d[2] * shape.d[3] * sizeof(float);
    for (int i = 0; i < images.size(); ++i) {
        images[i].create(shape.d[2], shape.d[3], CV_32FC3);

        std::vector<cv::cuda::GpuMat> channels {
            cv::cuda::GpuMat(shape.d[2], shape.d[3], CV_32F, static_cast<unsigned char*>(blobPtr) + 0 * width + shape.d[1] * width * i),
            cv::cuda::GpuMat(shape.d[2], shape.d[3], CV_32F, static_cast<unsigned char*>(blobPtr) + 1 * width + shape.d[1] * width * i),
            cv::cuda::GpuMat(shape.d[2], shape.d[3], CV_32F, static_cast<unsigned char*>(blobPtr) + 2 * width + shape.d[1] * width * i)
        };
        cv::cuda::merge(channels, images[i], stream);
    }
    return images;
}

cv::cuda::GpuMat trt::Img2Img::padRoi(const cv::cuda::GpuMat& input, const cv::Rect2i& roi, cv::cuda::Stream& stream) {
    int tl_x = roi.x;
    int tl_y = roi.y;
    int br_x = roi.x + roi.width;
    int br_y = roi.y + roi.height;
    int width = roi.width;
    int height = roi.height;

    if (tl_x < 0 || tl_y < 0 || br_x > input.cols || br_y > input.rows) {
        int left = 0, right = 0, top = 0, bottom = 0;

        if (tl_x < 0) {
            width += tl_x;
            left = -tl_x;
            tl_x = 0;
        }
        if (tl_y < 0) {
            height += tl_y;
            top = -tl_y;
            tl_y = 0;
        }
        if (br_x > input.cols) {
            width -= br_x - input.cols;
            right = br_x - input.cols;
        }
        if (br_y > input.rows) {
            height -= br_y - input.rows;
            bottom = br_y - input.rows;
        }

        cv::cuda::GpuMat output;
        cv::cuda::copyMakeBorder(input(cv::Rect2i(tl_x, tl_y, width, height)),
            output, top, bottom, left, right, cv::BORDER_REPLICATE, cv::Scalar(), stream);
        return output;
    } else {
        return input(cv::Rect2i(tl_x, tl_y, width, height));
    }
}

std::vector<cv::cuda::GpuMat> trt::Img2Img::generateTileWeights(const cv::Point2i& overlap, const cv::Size2i& size, cv::cuda::Stream& stream) {
    std::vector<cv::cuda::GpuMat> weights(4);
    weights[0] = cv::cuda::GpuMat(size, CV_32FC3, cv::Scalar(1.f, 1.f, 1.f));
    weights[3] = cv::cuda::GpuMat(size, CV_32FC3, cv::Scalar(1.f, 1.f, 1.f));

    // Top
    const int height = overlap.y + 1;
    for (int i = 1; i < height; ++i) {
        double alpha = static_cast<double>(i) / height;
        weights[0].row(i - 1).setTo(cv::Scalar(alpha, alpha, alpha), stream);
    }

    // Left
    const int width = overlap.x + 1;
    for (int i = 1; i < width; ++i) {
        double alpha = static_cast<double>(i) / width;
        weights[3].col(i - 1).setTo(cv::Scalar(alpha, alpha, alpha), stream);
    }

    // Bottom
    cv::cuda::flip(weights[0], weights[2], 0, stream);

    // Right
    cv::cuda::flip(weights[3], weights[1], 1, stream);

    return weights;
}

bool trt::Img2Img::serializeConfig(std::string& path, const BuildConfig& config) {
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

bool trt::Img2Img::deserializeConfig(const std::string& path, BuildConfig& config) {
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