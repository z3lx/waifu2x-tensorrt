#ifndef WAIFU2X_TENSORRT_UTILS_PATH_H
#define WAIFU2X_TENSORRT_UTILS_PATH_H

namespace utils {
    [[nodiscard]]
    [[maybe_unused]]
    static inline std::string getFileDirectory(const std::string& path) {
        size_t pos = path.find_last_of("/\\");
        if (pos == std::string::npos)
            return {};
        return path.substr(0, pos);
    }

    [[nodiscard]]
    [[maybe_unused]]
    static inline std::string getFileName(const std::string& path) {
        size_t pos = path.find_last_of("/\\");
        if (pos == std::string::npos)
            return path;
        return path.substr(pos + 1);
    }

    [[nodiscard]]
    [[maybe_unused]]
    static inline std::string getFileExtension(const std::string& path) {
        size_t pos = path.find_last_of('.');
        if (pos == std::string::npos)
            return {};
        return path.substr(pos + 1);
    }

    [[nodiscard]]
    [[maybe_unused]]
    static inline std::string removeFileExtension(const std::string& path) {
        size_t pos = path.find_last_of('.');
        if (pos == std::string::npos)
            return path;
        return path.substr(0, pos);
    }
}

#endif //WAIFU2X_TENSORRT_UTILS_PATH_H