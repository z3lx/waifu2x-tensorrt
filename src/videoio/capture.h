#ifndef WAIFU2X_TENSORRT_VIDEOIO_CAPTURE_H
#define WAIFU2X_TENSORRT_VIDEOIO_CAPTURE_H

#include <opencv2/core/mat.hpp>

class VideoCapture {
public:
    VideoCapture() noexcept;
    virtual ~VideoCapture();
    void open(const std::string& path);
    [[nodiscard]] bool isOpened() const noexcept;
    bool read(cv::Mat& frame);
    void release();

    // region Getters
    [[nodiscard]] const std::string& getFfmpegDir() const noexcept;
    [[nodiscard]] const cv::Size2i& getFrameSize() const noexcept;
    [[nodiscard]] double getFrameRate() const noexcept;
    [[nodiscard]] int getFrameCount() const noexcept;
    [[nodiscard]] int getFrameIndex() const noexcept;
    // endregion
private:
    FILE* pipe = nullptr;
    bool opened = false;

    std::string ffmpegDir;
    cv::Size2i frameSize = cv::Size2i(-1, -1);
    double frameRate = -1;
    int frameCount = -1;
    int frameIndex = -1;
};

#endif //WAIFU2X_TENSORRT_VIDEOIO_CAPTURE_H