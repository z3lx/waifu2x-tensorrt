#ifndef WAIFU2X_TENSORRT_VIDEOIO_CAPTURE_H
#define WAIFU2X_TENSORRT_VIDEOIO_CAPTURE_H

#include <opencv2/core/mat.hpp>

class VideoCapture {
public:
    VideoCapture();
    explicit VideoCapture(std::string ffmpegDir);
    virtual ~VideoCapture();
    void open(const std::string& path);
    bool isOpened() const;
    void read(cv::Mat& frame);
    void release();

private:
    std::string ffmpegDir;
    FILE* pipe;
    int frameWidth;
    int frameHeight;
    bool opened;
};

#endif //WAIFU2X_TENSORRT_VIDEOIO_CAPTURE_H