#include "img2img.h"
#include "utilities/time.h"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>

bool trt::Img2Img::render(cv::cuda::GpuMat& input, cv::cuda::GpuMat& output) try {
    // Allocate output
    output.create(input.rows * renderConfig.scaling.x, input.cols * renderConfig.scaling.y, CV_32FC3);
    output.setTo(cv::Scalar(0, 0, 0), stream);

    // Calculate tiles
    const cv::Rect2i inputRect = {0, 0, input.cols, input.rows};
    const cv::Rect2i outputRect = {0, 0, output.cols, output.rows};
    auto [tileCount, inputTileRects, outputTileRects] = calculateTiles(inputRect, outputRect);

    // Constants
    const auto tta = renderConfig.tta;
    const auto overlapping = renderConfig.overlap.x != 0 || renderConfig.overlap.y != 0;

    constexpr auto ttaSize = 8;
    const auto batchSize = renderConfig.batchSize;
    const auto stepsPerTile = tta ? ttaSize : 1;
    const auto batchCount = std::lround(std::ceil(static_cast<double>(tileCount * stepsPerTile) / batchSize));
    const auto stepCount = batchCount * batchSize;

    // Tile buffers and indices
    std::queue<std::tuple<int, int>> tileIndices;
    std::vector<cv::cuda::GpuMat> inputTiles;
    std::vector<cv::cuda::GpuMat> outputTiles;
    inputTiles.reserve(batchSize);
    outputTiles.reserve(batchSize);

    // Render image
    for (auto stepIndex = 0; stepIndex < stepCount; ++stepIndex) {
        const auto t0 = std::chrono::steady_clock::now();

        // Calculate indices
        auto tileIndex = stepIndex / stepsPerTile;
        auto augmentationIndex = stepIndex % stepsPerTile;
        auto batchIndex = stepIndex % batchSize;
        tileIndices.emplace(tileIndex, augmentationIndex);

        // Preprocess batch
        if (tileIndex < tileCount) {
            const auto inputTile = padRoi(input, inputTileRects[tileIndex], stream);
            if (tta && augmentationIndex != 0) {
                auto& ttaInputTile = ttaInputTiles[batchIndex];
                applyAugmentation(inputTile, ttaInputTile, inputTileSize, augmentationIndex);
                inputTiles.emplace_back(ttaInputTile);
            } else {
                inputTiles.emplace_back(inputTile);
            }
        } else {
            inputTiles.emplace_back(inputTileSize, CV_32FC3, cv::Scalar(0, 0, 0));
        }

        // Check if batch is full
        if (batchIndex != batchSize - 1)
            continue;

        // Infer batch
        if (!infer(inputTiles, outputTiles)) {
            logger.LOG(error, "Failed to infer tile " + std::to_string(tileIndex + 1)
                + "/" + std::to_string(tileCount) + ".");
            return false;
        }

        // Postprocess batch
        for (batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
            std::tie(tileIndex, augmentationIndex) = tileIndices.front();
            if (tileIndex == tileCount)
                break;
            tileIndices.pop();
            auto* outputTile = &outputTiles[batchIndex];
            auto& outputTileRect = outputTileRects[tileIndex];

            // Postprocess TTA
            if (tta) {
                if (augmentationIndex == 0) {
                    ttaOutputTile.setTo(cv::Scalar(0, 0, 0), stream);
                    cv::cuda::add(ttaOutputTile, *outputTile, ttaOutputTile, cv::noArray(), -1, stream);
                } else {
                    reverseAugmentation(*outputTile, tmpOutputMat, outputTileSize, augmentationIndex);
                    cv::cuda::add(ttaOutputTile, tmpOutputMat, ttaOutputTile, cv::noArray(), -1, stream);
                    if (augmentationIndex == ttaSize - 1) {
                        cv::cuda::multiply(ttaOutputTile, 1.0 / ttaSize, ttaOutputTile, 1, -1, stream);
                        outputTile = &tmpOutputMat;
                    }
                }
            }

            // Check if tile is fully rendered
            if (!(!tta || augmentationIndex == ttaSize - 1))
                continue;

            // Postprocess blending
            if (overlapping)
                applyBlending(*outputTile, *outputTile, outputTileRect, outputRect);

            // Add tile to output
            cv::cuda::add((*outputTile)(cv::Rect2i(0, 0, outputTileRect.width, outputTileRect.height)),
                output(outputTileRect), output(outputTileRect), cv::noArray(), -1, stream);
        }
        // Clear batch
        inputTiles.clear();

        // Log progress
        const auto t1 = std::chrono::steady_clock::now();
        const auto elapsed = utils::getElapsedMilliseconds(t0, t1);
        logger.LOG(info, "Rendered batch " + std::to_string(stepIndex / batchSize + 1) + "/" + std::to_string(batchCount)
            + " @ " + std::to_string(1000.0 / elapsed) + " it/s.");
    }

    // Postprocess output
    output.convertTo(output, CV_8UC3, 255.0, stream);
    cv::cuda::cvtColor(output, output, cv::COLOR_RGB2BGR, 0, stream);
    stream.waitForCompletion();

    return true;
}
catch (const std::exception& e) {
    logger.LOG(error, "Render failed unexpectedly: " + std::string(e.what()) + ".");
    return false;
}

std::tuple<const int, std::vector<cv::Rect2i>, std::vector<cv::Rect2i>> trt::Img2Img::calculateTiles(const cv::Rect2i& inputRect, const cv::Rect2i& outputRect) {
    // Calculate tiling
    const cv::Point2i tiling = {
        static_cast<int>(std::lround(std::ceil(static_cast<double>(inputRect.width - inputOverlap.x) / (scaledInputTileSize.width - inputOverlap.x)))),
        static_cast<int>(std::lround(std::ceil(static_cast<double>(inputRect.height - inputOverlap.y) / (scaledInputTileSize.height - inputOverlap.y))))
    };
    const auto tileCount = tiling.x * tiling.y;

    // Calculate rects
    std::vector<cv::Rect2i> inputTileRects;
    std::vector<cv::Rect2i> outputTileRects;
    inputTileRects.reserve(tileCount);
    outputTileRects.reserve(tileCount);

    for (auto i = 0; i < tiling.x; ++i) {
        for (auto j = 0; j < tiling.y; ++j) {
            // offset_border + offset_scaled_tile - offset_overlap
            inputTileRects.emplace_back(
                -((inputTileSize.width - scaledInputTileSize.width) / 2) + (i * scaledInputTileSize.width) - (i * inputOverlap.x),
                -((inputTileSize.height - scaledInputTileSize.height) / 2) + (j * scaledInputTileSize.height) - (j * inputOverlap.y),
                inputTileSize.width,
                inputTileSize.height
            );

            // offset_tile - offset_overlap
            const auto x = i * outputTileSize.width - (i * scaledOutputOverlap.x);
            const auto y = j * outputTileSize.height - (j * scaledOutputOverlap.y);
            outputTileRects.emplace_back(
                x,
                y,
                x + outputTileSize.width > outputRect.width ? outputRect.width - x : outputTileSize.width,
                y + outputTileSize.height > outputRect.height ? outputRect.height - y : outputTileSize.height
            );
        }
    }

    return std::make_tuple(tileCount, inputTileRects, outputTileRects);
}

cv::cuda::GpuMat trt::Img2Img::padRoi(const cv::cuda::GpuMat& input, const cv::Rect2i& roi, cv::cuda::Stream& stream) {
    int tl_x = roi.x;
    int tl_y = roi.y;
    int br_x = roi.x + roi.width;
    int br_y = roi.y + roi.height;
    int width = roi.width;
    int height = roi.height;

    if (tl_x < 0 || tl_y < 0 || br_x > input.cols || br_y > input.rows) {
        int left = 0, right = 0, top = 0, bottom = 0;

        if (tl_x < 0) {
            width += tl_x;
            left = -tl_x;
            tl_x = 0;
        }
        if (tl_y < 0) {
            height += tl_y;
            top = -tl_y;
            tl_y = 0;
        }
        if (br_x > input.cols) {
            width -= br_x - input.cols;
            right = br_x - input.cols;
        }
        if (br_y > input.rows) {
            height -= br_y - input.rows;
            bottom = br_y - input.rows;
        }

        cv::cuda::GpuMat output;
        cv::cuda::copyMakeBorder(input(cv::Rect2i(tl_x, tl_y, width, height)),
            output, top, bottom, left, right, cv::BORDER_REPLICATE, cv::Scalar(), stream);
        return output;
    } else {
        return input(cv::Rect2i(tl_x, tl_y, width, height));
    }
}

void trt::Img2Img::applyBlending(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, const cv::Rect2i& srcRect, const cv::Rect2i& dstRect) {
    if (srcRect.x > dstRect.x)
        cv::cuda::multiply(src, weights[3], dst, 1, -1, stream);

    if (srcRect.y > dstRect.y)
        cv::cuda::multiply(src, weights[0], dst, 1, -1, stream);

    if (srcRect.x + srcRect.width < dstRect.width)
        cv::cuda::multiply(src, weights[1], dst, 1, -1, stream);

    if (srcRect.y + srcRect.height < dstRect.height)
        cv::cuda::multiply(src, weights[2], dst, 1, -1, stream);
}

enum Augmentation {
    None,
    FlipHorizontal,
    FlipVertical,
    Rotate90,
    Rotate180,
    Rotate270,
    FlipHorizontalRotate90,
    FlipVerticalRotate90
};

void trt::Img2Img::applyAugmentation(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, const cv::Size2i& dstSize, int augmentationIndex) {
    switch (augmentationIndex) {
        default:
        case Augmentation::None:
            src.copyTo(dst, stream);
            break;

        case Augmentation::FlipHorizontal:
            cv::cuda::flip(src, dst, 0, stream);
            break;

        case Augmentation::FlipVertical:
            cv::cuda::flip(src, dst, 1, stream);
            break;

        case Augmentation::Rotate90:
            cv::cuda::rotate(src, dst, dstSize, 90,
                0, dstSize.height - 1, cv::INTER_NEAREST, stream);
            break;

        case Augmentation::Rotate180:
            cv::cuda::rotate(src, dst, dstSize, 180,
                dstSize.width - 1, dstSize.height - 1, cv::INTER_NEAREST, stream);
            break;

        case Augmentation::Rotate270:
            cv::cuda::rotate(src, dst, dstSize, 270,
                dstSize.width - 1, 0, cv::INTER_NEAREST, stream);
            break;

        case Augmentation::FlipHorizontalRotate90:
            cv::cuda::flip(src, tmpInputMat, 0, stream);
            cv::cuda::rotate(tmpInputMat, dst, dstSize, 90,
                0, dstSize.height - 1, cv::INTER_NEAREST, stream);
            break;

        case Augmentation::FlipVerticalRotate90:
            cv::cuda::flip(src, tmpInputMat, 1, stream);
            cv::cuda::rotate(tmpInputMat, dst, dstSize, 90,
                0, dstSize.height - 1, cv::INTER_NEAREST, stream);
            break;
    }
}

void trt::Img2Img::reverseAugmentation(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, const cv::Size2i& dstSize, int augmentationIndex) {
    switch (augmentationIndex) {
        default:
        case Augmentation::None:
            src.copyTo(dst, stream);
            break;

        case Augmentation::FlipHorizontal:
            cv::cuda::flip(src, dst, 0, stream);
            break;

        case Augmentation::FlipVertical:
            cv::cuda::flip(src, dst, 1, stream);
            break;

        case Augmentation::Rotate90:
            cv::cuda::rotate(src, dst, dstSize, 270,
                dstSize.width - 1, 0, cv::INTER_NEAREST, stream);
            break;

        case Augmentation::Rotate180:
            cv::cuda::rotate(src, dst, dstSize, 180,
                dstSize.width - 1, dstSize.height - 1, cv::INTER_NEAREST, stream);
            break;

        case Augmentation::Rotate270:
            cv::cuda::rotate(src, dst, dstSize, 90,
                0, dstSize.height - 1, cv::INTER_NEAREST, stream);
            break;

        case Augmentation::FlipHorizontalRotate90:
            cv::cuda::rotate(src, tmpOutputMat, dstSize, 270,
                dstSize.width - 1, 0, cv::INTER_NEAREST, stream);
            cv::cuda::flip(tmpOutputMat, dst, 0, stream);
            break;

        case Augmentation::FlipVerticalRotate90:
            cv::cuda::rotate(src, tmpOutputMat, dstSize, 270,
                dstSize.width - 1, 0, cv::INTER_NEAREST, stream);
            cv::cuda::flip(tmpOutputMat, dst, 1, stream);
            break;
    }
}