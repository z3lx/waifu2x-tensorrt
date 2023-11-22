#include "img2img.h"
#include "helper.h"

trt::Img2Img::Img2Img() = default;

trt::Img2Img::~Img2Img() {
    for (auto& buffer : buffers) {
        cudaAssert(cudaFree(buffer.first));
    }
}

void trt::Img2Img::setMessageCallback(MessageCallback callback) {
    logger.setMessageCallback(std::move(callback));
}

void trt::Img2Img::setProgressCallback(trt::ProgressCallback callback) {
    logger.setProgressCallback(std::move(callback));
}