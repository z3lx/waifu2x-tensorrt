#ifndef WAIFU2X_TENSORRT_VIDEOIO_WRITER_H
#define WAIFU2X_TENSORRT_VIDEOIO_WRITER_H

#include <cstdio>
#include <opencv2/core/mat.hpp>

class VideoWriter {
public:
    VideoWriter();
    virtual ~VideoWriter();
    void open();
    [[nodiscard]] bool isOpened() const;
    void write(const cv::Mat& frame);
    void release();

    // region Getters and setters
    [[nodiscard]] const std::string& getFfmpegDir() const;
    [[nodiscard]] double getFrameRate() const;
    [[nodiscard]] const cv::Size2i& getFrameSize() const;
    [[nodiscard]] const std::string& getOutputFile() const;
    [[nodiscard]] const std::string& getPixelFormat() const;
    [[nodiscard]] const std::string& getCodec() const;
    [[nodiscard]] int getQuality() const;

    VideoWriter& setFfmpegDir(const std::string& value);
    VideoWriter& setFrameRate(double value);
    VideoWriter& setFrameSize(const cv::Size2i& value);
    VideoWriter& setOutputFile(const std::string& value);
    VideoWriter& setPixelFormat(const std::string& value);
    VideoWriter& setCodec(const std::string& value);
    VideoWriter& setQuality(int value);
    // endregion

private:
    std::string ffmpegDir;
    double frameRate = -1;
    cv::Size2i frameSize = cv::Size2i(-1, -1);
    std::string outputFile;

    std::string pixelFormat = "yuv420p";
    std::string codec = "libx264";
    int quality = 23;
    // tune, preset, hardware accel...

    FILE* pipe = nullptr;
    bool opened = false;
};

#endif //WAIFU2X_TENSORRT_VIDEOIO_WRITER_H