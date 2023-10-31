#ifndef WAIFU2X_TENSORRT_VIDEOIO_WRITER_H
#define WAIFU2X_TENSORRT_VIDEOIO_WRITER_H

#include <cstdio>
#include <opencv2/core/mat.hpp>

class VideoWriter {
public:
    VideoWriter() noexcept;
    virtual ~VideoWriter() noexcept;
    void open();
    [[nodiscard]] bool isOpened() const noexcept;
    void write(const cv::Mat& frame);
    void release() noexcept;

    // region Getters and setters
    [[nodiscard]] const std::string& getFfmpegDir() const noexcept;
    [[nodiscard]] const cv::Size2i& getFrameSize() const noexcept;
    [[nodiscard]] double getFrameRate() const noexcept;
    [[nodiscard]] const std::string& getOutputFile() const noexcept;
    [[nodiscard]] const std::string& getPixelFormat() const noexcept;
    [[nodiscard]] const std::string& getCodec() const noexcept;
    [[nodiscard]] int getConstantRateFactor() const noexcept;
    [[nodiscard]] int getQuality() const noexcept;

    VideoWriter& setFfmpegDir(const std::string& value);
    VideoWriter& setFrameSize(const cv::Size2i& value);
    VideoWriter& setFrameRate(double value);
    VideoWriter& setOutputFile(const std::string& value);
    VideoWriter& setPixelFormat(const std::string& value);
    VideoWriter& setCodec(const std::string& value);
    VideoWriter& setConstantRateFactor(int value);
    VideoWriter& setQuality(int value);
    // endregion

private:
    FILE* pipe = nullptr;
    bool opened = false;

    std::string ffmpegDir;
    cv::Size2i frameSize = cv::Size2i(-1, -1);
    double frameRate = -1;
    std::string outputFile;
    std::string pixelFormat;
    std::string codec;
    int constantRateFactor = -1;
    int quality = -1;
    // tune, preset, hardware accel...
};

#endif //WAIFU2X_TENSORRT_VIDEOIO_WRITER_H