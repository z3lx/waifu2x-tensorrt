#include "img2img.h"

#include <utility>

#define LOG(severity, message) logger.log(severity, message, __FILE__, __FUNCTION__, __LINE__)
#define CUDASTREAM static_cast<cudaStream_t>(stream.cudaPtr())

trt::Img2Img::Img2Img() = default;

trt::Img2Img::~Img2Img() {
    for (auto& buffer : buffers) {
        cudaAssert(cudaFree(buffer.first));
    }
}

void trt::Img2Img::setLogCallback(LogCallback callback) {
    logger.setLogCallback(std::move(callback));
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
        LOG(error, "Failed to set cuda device to device id "
            + std::to_string(config.deviceId) + ": " + std::string(e.what()) + ".");
        return false;
    }

    // Create builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder) {
        LOG(error, "Failed to create infer builder.");
        return false;
    }

    // Create network
    auto flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flags));
    if (!network) {
        LOG(error, "Failed to create network.");
        return false;
    }

    // Create parser
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (!parser) {
        LOG(error, "Failed to create parser.");
        return false;
    }

    // Parse ONNX model
    auto parsed = parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kVERBOSE));
    if (!parsed) {
        LOG(error, "Failed to parse ONNX model.");
        return false;
    }

    // Create builder config
    auto builderConfig = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!builderConfig) {
        LOG(error, "Failed to create builder config.");
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
        LOG(error, "Failed to add optimization profile.");
        return false;
    }

    // Configure builder precision
    if (config.precision == Precision::FP16) {
        if (!builder->platformHasFastFp16()) {
            LOG(error, "Failed to set precision: platform does not support FP16");
            return false;
        }
        builderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else if (config.precision == Precision::TF32) {
        if (!builder->platformHasTf32()) {
            LOG(error, "Failed to set precision: platform does not support TF32");
            return false;
        }
        builderConfig->setFlag(nvinfer1::BuilderFlag::kTF32);
    }

    // Configure builder stream
    // Initialize stream before?
    builderConfig->setProfileStream(CUDASTREAM);

    // Build network
    std::unique_ptr<nvinfer1::IHostMemory> serializedNetwork {
        builder->buildSerializedNetwork(*network, *builderConfig)
    };
    if (!serializedNetwork) {
        LOG(error, "Failed to build serialized network.");
        return false;
    }

    // Serialize network
    try {
        std::string modelPath = onnxModelPath;
        serializeConfig(modelPath, config);
        std::ofstream engineFile(modelPath, std::ios::binary);
        engineFile.write(
            reinterpret_cast<const char*>(serializedNetwork->data()),
            static_cast<long long>(serializedNetwork->size())
        );
        engineFile.close();
    }
    catch (const std::exception& e) {
        LOG(error, "Failed to serialize network to disk: " + std::string(e.what()) + ".");
        return false;
    }

    return true;
}
catch (const std::exception& e) {
    LOG(error, "Engine build failed unexpectedly: " + std::string(e.what()) + ".");
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
        LOG(error, "Failed to set cuda device to device id "
            + std::to_string(config.deviceId) + ": " + std::string(e.what()) + ".");
        return false;
    }

    // Read engine
    std::vector<char> engineBuffer;
    std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG(error, "Failed to open engine file \"" + modelPath + "\".");
        return false;
    }
    std::streamsize fileSize = file.tellg();
    engineBuffer.resize(fileSize);
    file.seekg(0, std::ios::beg);
    file.read(engineBuffer.data(), fileSize);
    file.close();

    // Destroy existing engine
    if (context || engine || runtime) {
        context.reset();
        engine.reset();
        //runtime.reset();
    }

    // Create runtime if necessary
    if (!runtime) {
        runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
        if (!runtime) {
            LOG(error, "Failed to create infer runtime.");
            return false;
        }
    }

    // Deserialize engine
    engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(engineBuffer.data(), engineBuffer.size())
    );
    if (!engine) {
        LOG(error, "Failed to deserialize cuda engine from buffer.");
        return false;
    }

    // Validate engine
    const auto nbIOTensors = engine->getNbIOTensors();
    if (nbIOTensors != 2) {
        LOG(error, "Cuda engine has invalid number of IO tensors: "
            "expected 2, got " + std::to_string(nbIOTensors) + ".");
        return false;
    }
    for (int i = 0; i < nbIOTensors; ++i) {
        int nbDims = engine->getTensorShape(engine->getIOTensorName(i)).nbDims;
        if (nbDims != 4) {
            LOG(error, "Cuda engine has invalid IO tensor shape: "
                "expected 4 dims, got " + std::to_string(nbDims) + ".");
            return false;
        }
    }

    // Create execution context
    context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context) {
        LOG(error, "Failed to create execution context.");
        return false;
    }

    // Set tensor shapes
    nvinfer1::Dims4 inputShape{config.nbBatches, config.channels, config.height, config.width};
    if (!context->setInputShape(engine->getIOTensorName(0), inputShape)) {
        LOG(error, "Failed to set input tensor shape.");
        return false;
    }

    // Create stream
    stream = cv::cuda::Stream(cudaStreamNonBlocking);

    // Deallocate existing buffers
    if (!buffers.empty()) {
        try {
            for (auto& buffer : buffers) {
                cudaAssert(cudaFreeAsync(buffer.first, CUDASTREAM));
            }
        } catch (const std::exception& e) {
            LOG(error, "Failed to deallocate buffers: " + std::string(e.what()) + ".");
            return false;
        }
        buffers.clear();
    } else {
        buffers.reserve(nbIOTensors);
    }

    // Allocate buffers and set tensor addresses
    for (int i = 0; i < nbIOTensors; ++i) {
        auto tensorName = engine->getIOTensorName(i);
        auto tensorShape = context->getTensorShape(tensorName);
        auto tensorSize = tensorShape.d[0] * tensorShape.d[1] * tensorShape.d[2] * tensorShape.d[3];

        buffers.emplace_back(nullptr, tensorSize * sizeof(float));
        try {
            cudaAssert(cudaMallocAsync(&buffers[i].first, buffers[i].second, CUDASTREAM));
        }
        catch (const std::exception& e) {
            for (auto& buffer : buffers) {
                cudaFreeAsync(buffer.first, CUDASTREAM);
            }
            LOG(error, "Failed to allocate resources for tensor \""
                + std::string(tensorName) + "\": " + std::string(e.what()) + ".");
            return false;
        }

        if (!context->setTensorAddress(tensorName, buffers[i].first)) {
            LOG(error, "Failed to set tensor address for tensor \""
                + std::string(tensorName) + "\".");
            return false;
        }
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

    if (renderConfig.tta) {
        ttaInputTiles = std::vector<cv::cuda::GpuMat>(renderConfig.nbBatches);
        for (auto& ttaInputTile : ttaInputTiles) {
            ttaInputTile.create(inputTileSize, CV_32FC3);
        }
        ttaOutputTile.create(outputTileSize, CV_32FC3);
        tmpInputMat.create(inputTileSize, CV_32FC3);
        tmpOutputMat.create(outputTileSize, CV_32FC3);
    }

    return true;
}
catch (const std::exception& e) {
    LOG(error, "Engine load failed unexpectedly: " + std::string(e.what()) + ".");
    return false;
}

bool trt::Img2Img::render(cv::cuda::GpuMat& input, cv::cuda::GpuMat& output) try {
    std::chrono::time_point<std::chrono::steady_clock> t0, t1;
    double elapsed;

    output.create(input.rows * renderConfig.scaling.x, input.cols * renderConfig.scaling.y, CV_32FC3);
    output.setTo(cv::Scalar(0, 0, 0), stream);

    // Calculate tiling
    const cv::Point2i tiling = {
        static_cast<int>(std::lround(std::ceil(static_cast<double>(input.cols - inputOverlap.x) / (scaledInputTileSize.width - inputOverlap.x)))),
        static_cast<int>(std::lround(std::ceil(static_cast<double>(input.rows - inputOverlap.y) / (scaledInputTileSize.height - inputOverlap.y))))
    };
    const int tileCount = tiling.x * tiling.y;

    // Calculate tile rects
    std::vector<cv::Rect2i> inputTileRects;
    std::vector<cv::Rect2i> outputTileRects;
    inputTileRects.reserve(tileCount);
    outputTileRects.reserve(tileCount);

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

    const bool tta = renderConfig.tta;
    const bool overlapping = renderConfig.overlap.x != 0 || renderConfig.overlap.y != 0;

    const int batchSize = renderConfig.nbBatches;
    const int batchCount = std::lround(std::ceil(static_cast<double>(tileCount * (tta ? ttaSize : 1)) / batchSize));
    const int stepCount = batchCount * batchSize;
    const int stepsPerTile = tta ? ttaSize : 1;

    std::vector<cv::cuda::GpuMat> inputTiles;
    std::vector<cv::cuda::GpuMat> outputTiles;
    inputTiles.reserve(batchSize);
    outputTiles.reserve(batchSize);

    for (int stepIndex = 0; stepIndex < stepCount; ++stepIndex) {
        t0 = std::chrono::steady_clock::now();

        int tileIndex = stepIndex / stepsPerTile;
        int batchIndex = stepIndex % batchSize;
        int augmentationIndex = stepIndex % stepsPerTile;
        tileIndices.emplace(tileIndex, augmentationIndex);

        if (tileIndex < tileCount) {
            cv::cuda::GpuMat inputTile = padRoi(input, inputTileRects[tileIndex], stream);
            cv::cuda::GpuMat& ttaInputTile = ttaInputTiles[batchIndex];

            switch (augmentationIndex) {
                case 0:
                    inputTiles.emplace_back(inputTile);
                    break;

                case 1:
                    cv::cuda::flip(inputTile, ttaInputTile, 0, stream);
                    inputTiles.emplace_back(ttaInputTile);
                    break;

                case 2:
                    cv::cuda::flip(inputTile, ttaInputTile, 1, stream);
                    inputTiles.emplace_back(ttaInputTile);
                    break;

                case 3:
                    cv::cuda::flip(inputTile, ttaInputTile, -1, stream);
                    inputTiles.emplace_back(ttaInputTile);
                    break;

                case 4:
                    cv::cuda::rotate(inputTile, ttaInputTile, inputTileSize, 90, 0, inputTileSize.height - 1, cv::INTER_NEAREST, stream);
                    inputTiles.emplace_back(ttaInputTile);
                    break;

                case 5:
                    cv::cuda::rotate(inputTile, ttaInputTile, inputTileSize, -90, inputTileSize.width - 1, 0, cv::INTER_NEAREST, stream);
                    inputTiles.emplace_back(ttaInputTile);
                    break;

                case 6:
                    cv::cuda::rotate(inputTile, tmpInputMat, inputTileSize, 90, 0, inputTileSize.height - 1, cv::INTER_NEAREST, stream);
                    cv::cuda::flip(tmpInputMat, ttaInputTile, 0, stream);
                    inputTiles.emplace_back(ttaInputTile);
                    break;

                case 7:
                    cv::cuda::rotate(inputTile, tmpInputMat, inputTileSize, -90, inputTileSize.width - 1, 0, cv::INTER_NEAREST, stream);
                    cv::cuda::flip(tmpInputMat, ttaInputTile, 0, stream);
                    inputTiles.emplace_back(ttaInputTile);
                    break;

                default:
                    LOG(error, "Invalid TTA augmentation index: " + std::to_string(augmentationIndex) + ".");
                    return false;
            }
        } else {
            inputTiles.emplace_back(inputTileSize, CV_32FC3, cv::Scalar(0, 0, 0));
        }

        if (batchIndex == batchSize - 1) {
            if (!infer(inputTiles, outputTiles)) {
                LOG(error, "Failed to infer tile " + std::to_string(tileIndex + 1)
                    + "/" + std::to_string(tileCount) + ".");
                return false;
            }

            for (int i = 0; i < batchSize; ++i) {
                std::tie(tileIndex, augmentationIndex) = tileIndices.front();
                tileIndices.pop();

                if (tileIndex >= tileCount)
                    continue;

                cv::cuda::GpuMat* outputTile = &outputTiles[i];
                cv::Rect2i& outputTileRect = outputTileRects[tileIndex];

                if (tta) {
                    switch (augmentationIndex) {
                        case 0:
                            ttaOutputTile.setTo(cv::Scalar(0, 0, 0), stream);
                            cv::cuda::add(ttaOutputTile, *outputTile, ttaOutputTile, cv::noArray(), -1, stream);
                            break;

                        case 1:
                            cv::cuda::flip(*outputTile, tmpOutputMat, 0, stream);
                            cv::cuda::add(ttaOutputTile, tmpOutputMat, ttaOutputTile, cv::noArray(), -1, stream);
                            break;

                        case 2:
                            cv::cuda::flip(*outputTile, tmpOutputMat, 1, stream);
                            cv::cuda::add(ttaOutputTile, tmpOutputMat, ttaOutputTile, cv::noArray(), -1, stream);
                            break;

                        case 3:
                            cv::cuda::flip(*outputTile, tmpOutputMat, -1, stream);
                            cv::cuda::add(ttaOutputTile, tmpOutputMat, ttaOutputTile, cv::noArray(), -1, stream);
                            break;

                        case 4:
                            cv::cuda::rotate(*outputTile, tmpOutputMat, outputTileSize, -90, outputTileSize.width - 1, 0, cv::INTER_NEAREST, stream);
                            cv::cuda::add(ttaOutputTile, tmpOutputMat, ttaOutputTile, cv::noArray(), -1, stream);
                            break;

                        case 5:
                            cv::cuda::rotate(*outputTile, tmpOutputMat, outputTileSize, 90, 0, outputTileSize.height - 1, cv::INTER_NEAREST, stream);
                            cv::cuda::add(ttaOutputTile, tmpOutputMat, ttaOutputTile, cv::noArray(), -1, stream);
                            break;

                        case 6:
                            cv::cuda::flip(*outputTile, tmpOutputMat, 0, stream);
                            cv::cuda::rotate(tmpOutputMat, *outputTile, outputTileSize, -90, outputTileSize.width - 1, 0, cv::INTER_NEAREST, stream);
                            cv::cuda::add(ttaOutputTile, *outputTile, ttaOutputTile, cv::noArray(), -1, stream);
                            break;

                        case 7:
                            cv::cuda::flip(*outputTile, tmpOutputMat, 0, stream);
                            cv::cuda::rotate(tmpOutputMat, *outputTile, outputTileSize, 90, 0, outputTileSize.height - 1, cv::INTER_NEAREST, stream);
                            cv::cuda::add(ttaOutputTile, *outputTile, ttaOutputTile, cv::noArray(), -1, stream);
                            cv::cuda::divide(ttaOutputTile, cv::Scalar(ttaSize, ttaSize, ttaSize), ttaOutputTile, 1, -1, stream);
                            outputTile = &ttaOutputTile;
                            break;

                        default:
                            LOG(error, "Invalid TTA augmentation index: " + std::to_string(augmentationIndex) + ".");
                            return false;
                    }
                }

                if (!tta || augmentationIndex == ttaSize - 1) {
                    if (overlapping) {
                        if (outputTileRect.x != 0)
                            cv::cuda::multiply(*outputTile, weights[3], *outputTile, 1, -1, stream);

                        if (outputTileRect.y != 0)
                            cv::cuda::multiply(*outputTile, weights[0], *outputTile, 1, -1, stream);

                        if (outputTileRect.x + outputTileRect.width < output.cols)
                            cv::cuda::multiply(*outputTile, weights[1], *outputTile, 1, -1, stream);

                        if (outputTileRect.y + outputTileRect.height < output.rows)
                            cv::cuda::multiply(*outputTile, weights[2], *outputTile, 1, -1, stream);
                    }

                    cv::cuda::add((*outputTile)(cv::Rect2i(0, 0, outputTileRect.width, outputTileRect.height)),
                        output(outputTileRect), output(outputTileRect), cv::noArray(), -1, stream);
                }
            }

            t1 = std::chrono::steady_clock::now();
            elapsed = utils::getElapsedMilliseconds(t0, t1);
            LOG(info, "Rendered batch " + std::to_string(stepIndex / batchSize + 1) + "/" + std::to_string(batchCount)
                + " @ " + std::to_string(1000.0 / elapsed) + " it/s.");

            inputTiles.clear();
        }
    }

    output.convertTo(output, CV_8UC3, 255.0, stream);
    stream.waitForCompletion();

    return true;
}
catch (const std::exception& e) {
    LOG(error, "Render failed unexpectedly: " + std::string(e.what()) + ".");
    return false;
}

bool trt::Img2Img::infer(const std::vector<cv::cuda::GpuMat>& inputs, std::vector<cv::cuda::GpuMat>& outputs) try {
    // Check batch size
    if (inputs.size() != renderConfig.nbBatches) {
        LOG(error, "Input has invalid batch size: expected "
            + std::to_string(renderConfig.nbBatches) + ", got " + std::to_string(inputs.size()) + ".");
        return false;
    }

    // Check image size
    for (const auto& mat: inputs) {
        if (mat.channels() != renderConfig.channels) {
            LOG(error, "Input image has invalid number of channels: expected "
                + std::to_string(renderConfig.channels) + ", got " + std::to_string(mat.channels()) + ".");
            return false;
        }

        if (mat.rows != renderConfig.height) {
            LOG(error, "Input image has invalid height: expected "
                + std::to_string(renderConfig.height) + ", got " + std::to_string(mat.rows) + ".");
            return false;
        }

        if (mat.cols != renderConfig.width) {
            LOG(error, "Input image has invalid width: expected "
                + std::to_string(renderConfig.width) + ", got " + std::to_string(mat.cols) + ".");
            return false;
        }
    }

    // Preprocess input
    auto blob = blobFromImages(inputs, stream);

    // Copy blob to input tensor buffer
    cudaAssert(cudaMemcpyAsync(buffers[0].first, blob.ptr<void>(),
        buffers[0].second, cudaMemcpyDeviceToDevice, CUDASTREAM));

    // Enqueue inference
    if (!context->enqueueV3(CUDASTREAM)) {
        LOG(error, "Could not enqueue inference.");
        return false;
    }

    // Postprocess output
    outputs = imagesFromBlob(buffers[1].first, context->getTensorShape(engine->getIOTensorName(1)), stream);

    return true;
}
catch (const std::exception& e) {
    LOG(error, "Engine inference failed unexpectedly: " + std::string(e.what()) + ".");
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
        //LOG(error, "Invalid device index");
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
        //LOG(error, "Invalid engine.");
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
        //LOG(error, "Invalid device.");
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