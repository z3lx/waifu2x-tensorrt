#ifndef WAIFU2X_TENSORRT_TRT_CUDAHELPER_H
#define WAIFU2X_TENSORRT_TRT_CUDAHELPER_H

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include <NvInfer.h>

namespace trt {
    [[maybe_unused]]
    static inline void cudaAssert(cudaError_t error) {
        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }
    }

    [[maybe_unused]]
    [[nodiscard]]
    static inline std::vector<std::string> cudaGetDeviceNames() {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        std::vector<std::string> deviceNames;
        deviceNames.reserve(deviceCount);
        cudaDeviceProp deviceProp{};
        for (auto i = 0; i < deviceCount; ++i) {
            cudaGetDeviceProperties(&deviceProp, i);
            deviceNames.emplace_back(deviceProp.name);
        }
        return deviceNames;
    }

    [[maybe_unused]]
    [[nodiscard]]
    static inline std::string cudaGetDeviceName(int deviceId) {
        cudaDeviceProp deviceProp{};
        cudaGetDeviceProperties(&deviceProp, deviceId);
        return deviceProp.name;
    }

    [[maybe_unused]]
    [[nodiscard]]
    static inline int cudaGetDeviceId(const std::string& deviceName) {
        const auto deviceNames = cudaGetDeviceNames();
        const auto it = std::find(deviceNames.begin(), deviceNames.end(), deviceName);
        if (it != deviceNames.end())
            return static_cast<int>(std::distance(deviceNames.begin(), it));
        return -1;
    }
}

#endif //WAIFU2X_TENSORRT_TRT_CUDAHELPER_H