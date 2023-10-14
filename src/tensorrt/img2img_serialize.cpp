#include "img2img.h"
#include <nlohmann/json.hpp>
#include <fstream>

namespace trt {
    static void to_json(nlohmann::ordered_json& j, const BuildConfig& config) {

        j = nlohmann::ordered_json{
            {"deviceName", cudaGetDeviceName(config.deviceId)},
            {"precision", config.precision == Precision::FP16 ? "FP16" : "TF32"},
            {"minBatchSize", config.minBatchSize},
            {"optBatchSize", config.optBatchSize},
            {"maxBatchSize", config.maxBatchSize},
            {"minChannels", config.minChannels},
            {"optChannels", config.optChannels},
            {"maxChannels", config.maxChannels},
            {"minWidth", config.minWidth},
            {"optWidth", config.optWidth},
            {"maxWidth", config.maxWidth},
            {"minHeight", config.minHeight},
            {"optHeight", config.optHeight},
            {"maxHeight", config.maxHeight}
        };
    }

    static void from_json(const nlohmann::json& j, BuildConfig& config) {
        const auto deviceName = j.at("deviceName").get<std::string>();
        config.deviceId = cudaGetDeviceId(deviceName);
        const auto precision = j.at("precision").get<std::string>();
        config.precision = precision == "FP16" ? Precision::FP16 : Precision::TF32;
        j.at("minBatchSize").get_to(config.minBatchSize);
        j.at("optBatchSize").get_to(config.optBatchSize);
        j.at("maxBatchSize").get_to(config.maxBatchSize);
        j.at("minChannels").get_to(config.minChannels);
        j.at("optChannels").get_to(config.optChannels);
        j.at("maxChannels").get_to(config.maxChannels);
        j.at("minWidth").get_to(config.minWidth);
        j.at("optWidth").get_to(config.optWidth);
        j.at("maxWidth").get_to(config.maxWidth);
        j.at("minHeight").get_to(config.minHeight);
        j.at("optHeight").get_to(config.optHeight);
        j.at("maxHeight").get_to(config.maxHeight);
    }
}

bool trt::Img2Img::serializeConfig(const std::string& path, const BuildConfig& config) {
    nlohmann::ordered_json j = config;
    std::ofstream outputFile(path);
    if (!outputFile.is_open())
        return false;
    outputFile << std::setw(4) << j;
    return true;
}

bool trt::Img2Img::deserializeConfig(const std::string& path, trt::BuildConfig& config) {
    std::ifstream inputFile(path);
    if (!inputFile.is_open())
        return false;
    nlohmann::json j;
    inputFile >> j;
    config = j;
    return true;
}