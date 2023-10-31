#include "capture.h"
#include <stdexcept>
#include <filesystem>
#include <iostream>
#include <map>
#include <utility>

#if defined(_WIN32) || defined(_WIN64)
#define popen _popen
#define pclose _pclose
#endif

VideoCapture::VideoCapture() noexcept = default;

VideoCapture::~VideoCapture() {
    release();
}

std::map<std::string, std::string> parseKeyValueString(const std::string& input) {
    std::map<std::string, std::string> result;

    size_t startPos = 0;
    size_t endPos = 0;

    while (endPos != std::string::npos) {
        endPos = input.find('\n', startPos);
        const std::string line = input.substr(startPos, endPos - startPos);
        startPos = endPos + 1;

        const size_t equalsPos = line.find('=');
        if (equalsPos == std::string::npos)
            continue;
        const std::string key = line.substr(0, equalsPos);
        const std::string value = line.substr(equalsPos + 1);
        result[key] = value;
    }

    return result;
}

double fractionStringToDouble(const std::string& fraction) {
    std::istringstream ss(fraction);
    std::string numeratorString, denominatorString;
    if (std::getline(ss, numeratorString, '/') && std::getline(ss, denominatorString)) {
        const double numerator = std::stod(numeratorString);
        const double denominator = std::stod(denominatorString);
        if (denominator == 0)
            throw std::runtime_error("division by zero");
        return numerator / denominator;
    } else {
        throw std::invalid_argument("invalid fraction format");
    }
}

// TODO: ADD SUPPORT FOR ALPHA CHANNEL
// TODO: FIX FILE VALIDATION
void VideoCapture::open(const std::string& path) try {
    release();

    // Check if file exists
    if (!std::filesystem::exists(path))
        throw std::runtime_error("input file does not exist");

    // Get file info
    const auto ffprobeCmd = ffmpegDir +
        "ffprobe -v error -select_streams v:0 -show_entries "
        "stream=width,height,r_frame_rate,nb_frames "
        "-of default=noprint_wrappers=1 \"" + path + "\"";
    pipe = popen(ffprobeCmd.c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("could not open ffprobe with command"
            "\"" + ffprobeCmd + "\"");
    }

    char buffer[128];
    std::string output;
    output.reserve(128);
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
        output += buffer;
    pclose(pipe);
    pipe = nullptr;

    // Validate file
    std::transform(output.begin(), output.end(), output.begin(), ::tolower);
    if (output.find("invalid") != std::string::npos)
        throw std::runtime_error("input file is invalid");

    // Parse data
    const auto propMap = parseKeyValueString(output);
    frameSize.width = std::stoi(propMap.at("width"));
    frameSize.height = std::stoi(propMap.at("height"));
    frameRate = fractionStringToDouble(propMap.at("r_frame_rate"));
    frameCount = propMap.at("nb_frames") == "n/a" ? 1 : std::stoi(propMap.at("nb_frames"));

    // Open ffmpeg
    const auto ffmpegCmd = ffmpegDir +
        "ffmpeg -v error "
        "-i \"" + path + "\" -f image2pipe -vcodec rawvideo "
        "-pix_fmt bgr24 -";
    pipe = popen(ffmpegCmd.c_str(), "rb");
    if (!pipe) {
        throw std::runtime_error("could not open ffmpeg with command"
            "\"" + ffmpegCmd + "\"");
    }
    opened = true;
}
catch (const std::exception& e) {
    release();
    throw e;
}

bool VideoCapture::isOpened() const noexcept {
    return opened;
}

bool VideoCapture::read(cv::Mat& frame) {
    if (!opened)
        throw std::runtime_error("video capture is not opened");

    if (frameIndex + 1 >= frameCount)
        return false;

    frame.create(frameSize, CV_8UC3);
    if (fread(frame.data, 1, frame.total() * frame.elemSize(), pipe) <= 0)
        throw std::runtime_error("could not read frame from pipe");
    ++frameIndex;
    return true;
}

void VideoCapture::release() {
    if (pipe)
        _pclose(pipe);
    pipe = nullptr;
    opened = false;

    frameSize = cv::Size2i(-1, -1);
    frameRate = -1;
    frameCount = -1;
    frameIndex = -1;
}

#undef popen
#undef pclose

// region Getters
const std::string& VideoCapture::getFfmpegDir() const noexcept {
    return ffmpegDir;
}

const cv::Size2i& VideoCapture::getFrameSize() const noexcept {
    return frameSize;
}

double VideoCapture::getFrameRate() const noexcept {
    return frameRate;
}

int VideoCapture::getFrameCount() const noexcept {
    return frameCount;
}

int VideoCapture::getFrameIndex() const noexcept {
    return frameIndex;
}
// endregion