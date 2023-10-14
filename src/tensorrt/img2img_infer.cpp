#include "img2img.h"
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>

bool trt::Img2Img::infer(const std::vector<cv::cuda::GpuMat>& inputs, std::vector<cv::cuda::GpuMat>& outputs) try {
    // Check batch size
    if (inputs.size() != renderConfig.batchSize) {
        logger.LOG(error, "Input has invalid batch size: expected "
            + std::to_string(renderConfig.batchSize) + ", got " + std::to_string(inputs.size()) + ".");
        return false;
    }

    // Check image size
    for (const auto& mat: inputs) {
        if (mat.channels() != renderConfig.channels) {
            logger.LOG(error, "Input image has invalid number of channels: expected "
                + std::to_string(renderConfig.channels) + ", got " + std::to_string(mat.channels()) + ".");
            return false;
        }

        if (mat.rows != renderConfig.height) {
            logger.LOG(error, "Input image has invalid height: expected "
                + std::to_string(renderConfig.height) + ", got " + std::to_string(mat.rows) + ".");
            return false;
        }

        if (mat.cols != renderConfig.width) {
            logger.LOG(error, "Input image has invalid width: expected "
                + std::to_string(renderConfig.width) + ", got " + std::to_string(mat.cols) + ".");
            return false;
        }
    }

    const auto& cudaStream = cudaGetCudaStream(stream);

    // Preprocess input
    auto blob = blobFromImages(inputs, stream);

    // Copy blob to input tensor buffer
    cudaAssert(cudaMemcpyAsync(buffers[0].first, blob.ptr<void>(),
        buffers[0].second, cudaMemcpyDeviceToDevice, cudaStream));

    // Enqueue inference
    if (!context->enqueueV3(cudaStream)) {
        logger.LOG(error, "Could not enqueue inference.");
        return false;
    }

    // Postprocess output
    outputs = imagesFromBlob(buffers[1].first, context->getTensorShape(engine->getIOTensorName(1)), stream);

    return true;
}
catch (const std::exception& e) {
    logger.LOG(error, "Engine inference failed unexpectedly: " + std::string(e.what()) + ".");
    return false;
}

cv::cuda::GpuMat trt::Img2Img::blobFromImages(const std::vector<cv::cuda::GpuMat>& images, cv::cuda::Stream& stream) {
    cv::cuda::GpuMat blob(static_cast<int>(images.size()), images[0].channels() * images[0].rows * images[0].cols, CV_8U);

    size_t width = images[0].cols * images[0].rows;
    for (auto i = 0; i < images.size(); ++i) {
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

    size_t width = shape.d[2] * shape.d[3];
    for (auto i = 0; i < images.size(); ++i) {
        images[i].create(shape.d[2], shape.d[3], CV_32FC3);

        std::vector<cv::cuda::GpuMat> channels {
            cv::cuda::GpuMat(shape.d[2], shape.d[3], CV_32F, static_cast<float*>(blobPtr) + 0 * width + shape.d[1] * width * i),
            cv::cuda::GpuMat(shape.d[2], shape.d[3], CV_32F, static_cast<float*>(blobPtr) + 1 * width + shape.d[1] * width * i),
            cv::cuda::GpuMat(shape.d[2], shape.d[3], CV_32F, static_cast<float*>(blobPtr) + 2 * width + shape.d[1] * width * i)
        };

        cv::cuda::merge(channels, images[i], stream);
    }
    return images;
}