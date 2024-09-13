#include <stdexcept>
#include "writer.h"

#if defined(_WIN32) || defined(_WIN64)
#define popen _popen
#define pclose _pclose
#endif

VideoWriter::VideoWriter() = default;

VideoWriter::~VideoWriter() {
    release();
}

void VideoWriter::open() {
    release();

    if (frameRate <= 0)
        throw std::invalid_argument("frame rate must be greater than 0");

    if (frameSize.width <= 0 || frameSize.height <= 0)
        throw std::invalid_argument("frame size must be greater than 0");

    std::string ffmpegCmd = "ffmpeg -v error -y -f rawvideo -vcodec rawvideo"
        " -s " + std::to_string(frameSize.width) + "x" + std::to_string(frameSize.height) +
        " -pix_fmt bgr24"
        " -r " + std::to_string(frameRate) +
        " -i -"
        " -vcodec " + codec +
        " -pix_fmt " + pixelFormat +
        " -crf " + std::to_string(quality) +
        " \"" + outputFile + "\"";

    pipe = popen(ffmpegCmd.c_str(), "wb");
    if (!pipe)
        throw std::runtime_error("could not open ffmpeg pipe");
    opened = true;
}

bool VideoWriter::isOpened() const {
    return opened;
}

void VideoWriter::write(const cv::Mat& frame) {
    if (!opened)
        throw std::runtime_error("video writer is not opened");

    if (frame.size() != frameSize)
        throw std::invalid_argument("frame size does not match");

    if (frame.type() != CV_8UC3)
        throw std::invalid_argument("frame type must be CV_8UC3");

    if (fwrite(frame.data, 1, frame.total() * frame.elemSize(), pipe) <= 0)
        throw std::runtime_error("could not write frame to pipe");
}

void VideoWriter::release() {
    if (pipe)
        pclose(pipe);
    pipe = nullptr;
    opened = false;
}

// region Getters and setters
const std::string& VideoWriter::getFfmpegDir() const {
    return ffmpegDir;
}

const std::string& VideoWriter::getPixelFormat() const {
    return pixelFormat;
}

const std::string& VideoWriter::getCodec() const {
    return codec;
}

int VideoWriter::getQuality() const {
    return quality;
}

double VideoWriter::getFrameRate() const {
    return frameRate;
}

const cv::Size2i& VideoWriter::getFrameSize() const {
    return frameSize;
}

const std::string& VideoWriter::getOutputFile() const {
    return outputFile;
}

constexpr auto errorWriterOpened = "properties cannot be set when writer is open";

VideoWriter& VideoWriter::setFfmpegDir(const std::string& value) {
    if (opened)
        throw std::runtime_error(errorWriterOpened);
    ffmpegDir = value;
    return *this;
}

VideoWriter& VideoWriter::setPixelFormat(const std::string& value) {
    if (opened)
        throw std::runtime_error(errorWriterOpened);
    pixelFormat = value;
    return *this;
}

VideoWriter& VideoWriter::setCodec(const std::string& value) {
    if (opened)
        throw std::runtime_error(errorWriterOpened);
    codec = value;
    return *this;
}

VideoWriter& VideoWriter::setQuality(int value) {
    if (opened)
        throw std::runtime_error(errorWriterOpened);
    if (value < 0 || value > 51)
        throw std::invalid_argument("quality must be between 0 and 51");
    quality = value;
    return *this;
}

VideoWriter& VideoWriter::setFrameRate(double value) {
    if (opened)
        throw std::runtime_error(errorWriterOpened);
    if (value <= 0)
        throw std::invalid_argument("frame rate must be greater than 0");
    frameRate = value;
    return *this;
}

VideoWriter& VideoWriter::setFrameSize(const cv::Size2i& value) {
    if (opened)
        throw std::runtime_error(errorWriterOpened);
    if (value.width <= 0 || value.height <= 0)
        throw std::invalid_argument("frame size must be greater than 0");
    frameSize = value;
    return *this;
}

VideoWriter& VideoWriter::setOutputFile(const std::string& value) {
    if (opened)
        throw std::runtime_error(errorWriterOpened);
    outputFile = value;
    return *this;
}
// endregion

#undef popen
#undef pclose