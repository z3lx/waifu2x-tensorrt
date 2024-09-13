#ifndef WAIFU2X_TENSORRT_UTILS_PATH_H
#define WAIFU2X_TENSORRT_UTILS_PATH_H

#include <filesystem>

namespace utils {
    static inline std::vector<std::filesystem::path> findFilesByExtension(
        const std::vector<std::filesystem::path>& paths,
        const std::vector<std::string>& extensions,
        bool recursive = false) {
        std::vector<std::filesystem::path> filePaths;

        auto match = [&](const std::filesystem::path& path) {
            if (!path.has_extension())
                return;

            if (std::find(extensions.begin(), extensions.end(), path.extension().string()) != extensions.end())
                filePaths.emplace_back(path);
        };

        for (const auto& path : paths) {
            if (std::filesystem::is_regular_file(path))
                match(path);

            else if (std::filesystem::is_directory(path)) {
                if (recursive) {
                    for (const auto& entry: std::filesystem::recursive_directory_iterator(path))
                        match(entry.path());
                } else {
                    for (const auto& entry: std::filesystem::directory_iterator(path))
                        match(entry.path());
                }
            }
        }

        return filePaths;
    }
}

#endif //WAIFU2X_TENSORRT_UTILS_PATH_H