#include "img2img.h"
#include "utilities/path.h"
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <filesystem>
#include <fstream>

// TODO: SET OPTIMIZATION PROFILE VIA nvinfer1::IExecutionContext::setOptimizationProfileAsync
bool trt::Img2Img::load(const std::string& modelPath, trt::RenderConfig& config) try {
    // Find engine
    std::string enginePath;
    try {
        enginePath = getEnginePath(modelPath, config);
    } catch (const std::exception& e) {
        logger.LOG(error, "Failed to find engine file for model \"" + modelPath + "\": " + std::string(e.what()) + ".");
        return false;
    }

    // Set cuda device
    try {
        trt::cudaAssert(cudaSetDevice(config.deviceId));
    }
    catch (const std::exception& e) {
        logger.LOG(error, "Failed to set cuda device to device id "
            + std::to_string(config.deviceId) + ": " + std::string(e.what()) + ".");
        return false;
    }

    // Read engine
    std::vector<char> engineBuffer;
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        logger.LOG(error, "Failed to open engine file \"" + enginePath + "\".");
        return false;
    }
    std::streamsize fileSize = file.tellg();
    engineBuffer.resize(fileSize);
    file.seekg(0, std::ios::beg);
    file.read(engineBuffer.data(), fileSize);

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
            logger.LOG(error, "Failed to create infer runtime.");
            return false;
        }
    }

    // Deserialize engine
    engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(engineBuffer.data(), engineBuffer.size())
    );
    if (!engine) {
        logger.LOG(error, "Failed to deserialize cuda engine from buffer.");
        return false;
    }

    // Validate engine
    const auto nbIOTensors = engine->getNbIOTensors();
    if (nbIOTensors != 2) {
        logger.LOG(error, "Cuda engine has invalid number of IO tensors: "
            "expected 2, got " + std::to_string(nbIOTensors) + ".");
        return false;
    }
    for (int i = 0; i < nbIOTensors; ++i) {
        int nbDims = engine->getTensorShape(engine->getIOTensorName(i)).nbDims;
        if (nbDims != 4) {
            logger.LOG(error, "Cuda engine has invalid IO tensor shape: "
                "expected 4 dims, got " + std::to_string(nbDims) + ".");
            return false;
        }
    }

    // Create execution context
    context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context) {
        logger.LOG(error, "Failed to create execution context.");
        return false;
    }

    // Set tensor shapes
    nvinfer1::Dims4 inputShape{config.batchSize, config.channels, config.height, config.width};
    if (!context->setInputShape(engine->getIOTensorName(0), inputShape)) {
        logger.LOG(error, "Failed to set input tensor shape.");
        return false;
    }

    // Create stream
    stream = cv::cuda::Stream(cudaStreamNonBlocking);
    const auto& cudaStream = cudaGetCudaStream(stream);

    // Deallocate existing buffers
    if (!buffers.empty()) {
        try {
            for (auto& buffer : buffers) {
                cudaAssert(cudaFreeAsync(buffer.first, cudaStream));
            }
        } catch (const std::exception& e) {
            logger.LOG(error, "Failed to deallocate buffers: " + std::string(e.what()) + ".");
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
            cudaAssert(cudaMallocAsync(&buffers[i].first, buffers[i].second, cudaStream));
        }
        catch (const std::exception& e) {
            for (auto& buffer : buffers) {
                cudaFreeAsync(buffer.first, cudaStream);
            }
            logger.LOG(error, "Failed to allocate resources for tensor \""
                + std::string(tensorName) + "\": " + std::string(e.what()) + ".");
            return false;
        }

        if (!context->setTensorAddress(tensorName, buffers[i].first)) {
            logger.LOG(error, "Failed to set tensor address for tensor \""
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

    createTileWeights(weights, scaledOutputOverlap, outputTileSize, stream);

    if (renderConfig.tta) {
        ttaInputTiles.resize(renderConfig.batchSize);
        for (auto& ttaInputTile : ttaInputTiles) {
            ttaInputTile.create(inputTileSize, CV_32FC3);
        }
        ttaOutputTile.create(outputTileSize, CV_32FC3);
        tmpInputMat.create(inputTileSize, CV_32FC3);
        tmpOutputMat.create(outputTileSize, CV_32FC3);
    } else {
        ttaInputTiles.clear();
        ttaOutputTile.release();
        tmpInputMat.release();
        tmpOutputMat.release();
    }

    return true;
}
catch (const std::exception& e) {
    logger.LOG(error, "Engine load failed unexpectedly: " + std::string(e.what()) + ".");
    return false;
}

std::string trt::Img2Img::getEnginePath(const std::string& modelPath, const trt::RenderConfig& config) {
    namespace fs = std::filesystem;
    if (!fs::exists(modelPath))
        throw std::runtime_error("model file does not exist");

    std::string engineName = fs::path(modelPath).stem().string();
    std::string enginePath;
    for (const auto& entry : fs::directory_iterator(fs::path(modelPath).parent_path())) {
        if (!entry.is_regular_file())
            continue;

        const auto& path = entry.path();
        if (path.filename().string().rfind(engineName, 0) != 0 ||
            path.extension().string() != ".trt")
            continue;

        std::string configPath = utils::removeFileExtension(path.string()) + ".json";
        if (!fs::exists(configPath))
            continue;

        BuildConfig buildConfig;
        deserializeConfig(configPath, buildConfig);
        if (isCompatible(config, buildConfig)) {
            if (isOptimized(config, buildConfig)) {
                enginePath = path.string();
                break;
            } else if (enginePath.empty()) {
                enginePath = path.string();
            }
        }
    }

    if (enginePath.empty())
        throw std::runtime_error("could not satisfy render configuration");
    return enginePath;
}

bool trt::Img2Img::isCompatible(const trt::RenderConfig& renderConfig, const trt::BuildConfig& buildConfig) {
    return renderConfig.deviceId == buildConfig.deviceId &&
        renderConfig.precision == buildConfig.precision &&
        renderConfig.batchSize >= buildConfig.minBatchSize &&
        renderConfig.batchSize <= buildConfig.maxBatchSize &&
        renderConfig.channels >= buildConfig.minChannels &&
        renderConfig.channels <= buildConfig.maxChannels &&
        renderConfig.width >= buildConfig.minWidth &&
        renderConfig.width <= buildConfig.maxWidth &&
        renderConfig.height >= buildConfig.minHeight &&
        renderConfig.height <= buildConfig.maxHeight;
}

bool trt::Img2Img::isOptimized(const trt::RenderConfig& renderConfig, const trt::BuildConfig& buildConfig) {
    return renderConfig.batchSize == buildConfig.optBatchSize &&
        renderConfig.channels == buildConfig.optChannels &&
        renderConfig.width == buildConfig.optWidth &&
        renderConfig.height == buildConfig.optHeight;
}

void trt::Img2Img::createTileWeights(std::array<cv::cuda::GpuMat, 4>& weights, const cv::Point2i& overlap, const cv::Size2i& size, cv::cuda::Stream& stream) {
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
}

