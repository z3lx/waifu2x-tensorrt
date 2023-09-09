#ifndef WAIFU2X_TENSORRT_TRT_HELPER_H
#define WAIFU2X_TENSORRT_TRT_HELPER_H

#include <NvInfer.h>
#include <stdexcept>

namespace trt {
    inline void cudaAssert(cudaError_t error) {
        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }
    }
}

#endif //WAIFU2X_TENSORRT_TRT_HELPER_H