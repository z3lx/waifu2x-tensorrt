#include <stdexcept>
#include "writer.h"

#if defined(_WIN32) || defined(_WIN64)
#define popen _popen
#define pclose _pclose
#endif

VideoWriter::VideoWriter() noexcept = default;

VideoWriter::~VideoWriter() noexcept {
    release();
}

void VideoWriter::open() {
    release();

    if (frameSize.width <= 0 || frameSize.height <= 0)
        throw std::invalid_argument("frame size must be greater than 0");

    if (outputFile.empty())
        throw std::invalid_argument("output file is empty");

    std::string ffmpegCmd = "ffmpeg -v error -y -f rawvideo -vcodec rawvideo"
        " -s " + std::to_string(frameSize.width) + "x" + std::to_string(frameSize.height) +
        " -pix_fmt bgr24" +
        (frameRate <= 0 ? "" : " -r " + std::to_string(frameRate)) +
        " -i -" +
        (codec.empty() ? "" : " -vcodec " + codec) +
        (pixelFormat.empty() ? "" : " -pix_fmt " + pixelFormat) +
        (constantRateFactor < 0 ? "" : " -crf " + std::to_string(constantRateFactor)) +
        (quality < 0 ? "" : " -q:v " + std::to_string(quality)) +
        " \"" + outputFile + "\"";

    pipe = popen(ffmpegCmd.c_str(), "wb");
    if (!pipe)
        throw std::runtime_error("could not open ffmpeg pipe");
    opened = true;
}

bool VideoWriter::isOpened() const noexcept {
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

void VideoWriter::release() noexcept {
    if (pipe)
        pclose(pipe);
    pipe = nullptr;
    opened = false;

    frameRate = -1;
    frameSize = cv::Size2i(-1, -1);
    outputFile = "";
    pixelFormat = "";
    codec = "";
    constantRateFactor = -1;
    quality = -1;
}

// region Getters and setters
const std::string& VideoWriter::getFfmpegDir() const noexcept {
    return ffmpegDir;
}

const cv::Size2i& VideoWriter::getFrameSize() const noexcept {
    return frameSize;
}

double VideoWriter::getFrameRate() const noexcept {
    return frameRate;
}

const std::string& VideoWriter::getOutputFile() const noexcept {
    return outputFile;
}

const std::string& VideoWriter::getPixelFormat() const noexcept {
    return pixelFormat;
}

const std::string& VideoWriter::getCodec() const noexcept {
    return codec;
}

int VideoWriter::getConstantRateFactor() const noexcept {
    return constantRateFactor;
}

int VideoWriter::getQuality() const noexcept {
    return quality;
}

constexpr auto errorWriterOpened = "properties cannot be set when writer is open";

VideoWriter& VideoWriter::setFfmpegDir(const std::string& value) {
    if (opened)
        throw std::runtime_error(errorWriterOpened);
    ffmpegDir = value;
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

VideoWriter& VideoWriter::setFrameRate(double value) {
    if (opened)
        throw std::runtime_error(errorWriterOpened);
    if (value <= 0)
        throw std::invalid_argument("frame rate must be greater than 0");
    frameRate = value;
    return *this;
}

VideoWriter& VideoWriter::setOutputFile(const std::string& value) {
    if (opened)
        throw std::runtime_error(errorWriterOpened);
    outputFile = value;
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

VideoWriter& VideoWriter::setConstantRateFactor(int value) {
    if (opened)
        throw std::runtime_error(errorWriterOpened);
    if (value > 51)
        throw std::invalid_argument("constant rate factor must be between 0 and 51");
    constantRateFactor = value;
    return *this;
}

VideoWriter& VideoWriter::setQuality(int value) {
    if (opened)
        throw std::runtime_error(errorWriterOpened);
    if (value == 0 || value > 31)
        throw std::invalid_argument("quality must be between 1 and 31");
    quality = value;
    return *this;
}
// endregion

#undef popen
#undef pclose