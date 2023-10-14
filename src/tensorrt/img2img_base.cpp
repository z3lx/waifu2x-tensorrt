#include "img2img.h"
#include "helper.h"

trt::Img2Img::Img2Img() = default;

trt::Img2Img::~Img2Img() {
    for (auto& buffer : buffers) {
        cudaAssert(cudaFree(buffer.first));
    }
}

void trt::Img2Img::setLogCallback(LogCallback callback) {
    logger.setLogCallback(std::move(callback));
}