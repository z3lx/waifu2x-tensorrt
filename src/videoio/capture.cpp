#include "capture.h"
#include <stdexcept>
#include <filesystem>
#include <iostream>
#include <map>

VideoCapture::VideoCapture()
    : ffmpegDir("") {

}

VideoCapture::VideoCapture(std::string ffmpegDir)
    : ffmpegDir(std::move(ffmpegDir)) {

}

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

// TODO: ADD SUPPORT FOR ALPHA CHANNEL
// TODO: FIX IMAGE VALIDATION
void VideoCapture::open(const std::string& path) try {
    release();

    // Check if ffmpeg and ffprobe exist
    auto ffmpegCmd = ffmpegDir + "ffmpeg -version";
    auto ffprobeCmd = ffmpegDir + "ffprobe -version";
    if (_popen(ffmpegCmd.c_str(), "r") == nullptr ||
        _popen(ffprobeCmd.c_str(), "r") == nullptr) {
        throw std::runtime_error("ffmpeg or ffprobe not found");
    }

    // Check if file exists
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("input file does not exist");
    }

    // Get file info
    ffprobeCmd = ffmpegDir +
        "ffprobe -v error -select_streams v:0 -show_entries "
        "stream=width,height,r_frame_rate,nb_frames "
        "-of default=noprint_wrappers=1 \"" + path + "\"";
    pipe = _popen(ffprobeCmd.c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("could not open ffprobe with command"
            "\"" + ffprobeCmd + "\"");
    }

    char buffer[128];
    std::string output;
    output.reserve(128);
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
        output += buffer;
    _pclose(pipe);
    pipe = nullptr;

    std::transform(output.begin(), output.end(), output.begin(), ::tolower);
    if (output.find("invalid") != std::string::npos) {
        throw std::runtime_error("input file is invalid");
    }

    // Parse info
    const auto propMap = parseKeyValueString(output);
    frameWidth = std::stoi(propMap.at("width"));
    frameHeight = std::stoi(propMap.at("height"));

    // Open ffmpeg
    ffmpegCmd = ffmpegDir +
        "ffmpeg -i \"" + path + "\" -f image2pipe -vcodec rawvideo "
        "-pix_fmt bgr24 -";
    pipe = _popen(ffmpegCmd.c_str(), "rb");
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

bool VideoCapture::isOpened() const {
    return opened;
}

void VideoCapture::read(cv::Mat& frame) {
    if (!opened)
        throw std::runtime_error("video capture is not opened");

    frame.create(frameHeight, frameWidth, CV_8UC3);
    if (fread(frame.data, 1, frame.total() * frame.elemSize(), pipe) <= 0)
        throw std::runtime_error("could not read frame from pipe");
}

void VideoCapture::release() {
    if (pipe)
        _pclose(pipe);
    pipe = nullptr;
    opened = false;
}