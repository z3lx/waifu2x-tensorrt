#ifndef WAIFU2X_TENSORRT_UTILS_TIME_H
#define WAIFU2X_TENSORRT_UTILS_TIME_H

#include <chrono>

namespace utils {
    template<typename T>
    inline double getElapsedMilliseconds(std::chrono::time_point<T> t0, std::chrono::time_point<T> t1) {
        return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / 1000.0;
    }
}

#endif //WAIFU2X_TENSORRT_UTILS_TIME_H