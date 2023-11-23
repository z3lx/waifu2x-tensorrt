#include "tensorrt/img2img.h"
#include "utilities/path.h"
#include "videoio/capture.h"
#include "videoio/writer.h"
#include <CLI/CLI.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

#define SPDLOG_LEVEL_NAMES { "TRACE", "DEBUG", "INFO ", "WARN ", "ERROR", "FATAL", "OFF" }
#include <spdlog/sinks/stdout_color_sinks.h>

int main(int argc, char *argv[]) {
    auto console = spdlog::stdout_color_mt("console");
    console->set_level(spdlog::level::info);
    console->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");

    // region Argument Parser
    CLI::App app("waifu2x-tensorrt");
    app.fallthrough()
        ->require_subcommand(1)
        ->get_formatter()->column_width(64);
    app.set_help_flag("", "");
    app.set_help_all_flag("-h, --help", "Print this help message and exit");

    std::string model;
    const auto modelChoices = {
        "cunet/art",
        "swin_unet/art",
        "swin_unet/art_scan",
        "swin_unet/photo"
    };
    app.add_option("--model", model)
        ->description("Set the model to use")
        ->check(CLI::IsMember(modelChoices))
        ->required();

    int scale;
    const auto scaleChoices = {
        1, 2, 4
    };
    app.add_option("--scale", scale)
        ->description("Set the scale factor")
        ->check(CLI::IsMember(scaleChoices))
        ->required();

    int noise;
    const auto noiseChoices = {
        -1, 0, 1, 2, 3
    };
    app.add_option("--noise", noise)
        ->description("Set the noise level")
        ->check(CLI::IsMember(noiseChoices))
        ->required();

    int batchSize;
    app.add_option("--batchSize", batchSize)
        ->description("Set the batch size")
        ->check(CLI::PositiveNumber)
        ->required();

    int tileSize;
    const auto tileSizeChoices = {
        64, 256, 400, 640
    };
    app.add_option("--tileSize", tileSize)
        ->description("Set the tile size")
        ->check(CLI::IsMember(tileSizeChoices))
        ->required();

    int deviceId = 0;
    app.add_option("--device", deviceId)
        ->description("Set the GPU device ID")
        ->default_val(deviceId)
        ->check(CLI::NonNegativeNumber);

    trt::Precision precision = trt::Precision::FP16;
    const std::map<std::string, trt::Precision> precisionMap = {
        {"fp16", trt::Precision::FP16},
        {"tf32", trt::Precision::TF32}
    };
    app.add_option("--precision", precision)
        ->description("Set the precision")
        ->default_val(precision)
        ->transform(CLI::CheckedTransformer(precisionMap, CLI::ignore_case));

    auto render = app.add_subcommand("render", "Render image(s)/video(s)");

    std::vector<std::filesystem::path> inputPaths;
    render->add_option("-i, --input", inputPaths)
        ->description("Set the input paths")
        ->check(CLI::ExistingPath)
        ->required();

    bool recursive = false;
    render->add_flag("--recursive", recursive)
        ->description("Search for input files recursively");

    std::filesystem::path outputDirectory;
    render->add_option("-o, --output", outputDirectory)
        ->description("Set the output directory")
        ->check(CLI::ExistingDirectory);

    double blend = 1.0/16.0;
    const auto blendChoices = {
        1.0/8.0, 1.0/16.0, 1.0/32.0, 0.0
    };
    render->add_option("--blend", blend)
        ->description("Set the percentage of overlap between two tiles to blend")
        ->default_val(blend)
        ->check(CLI::IsMember(blendChoices));

    bool tta = false;
    render->add_flag("--tta", tta)
        ->description("Enable test-time augmentation")
        ->default_val(tta);

    std::string codec = "libx264";
    render->add_option("--codec", codec)
        ->description("Set the codec (video only)")
        ->default_val(codec);

    std::string pixelFormat = "yuv420p";
    render->add_option("--pix_fmt", pixelFormat)
        ->description("Set the pixel format (video only)")
        ->default_val(pixelFormat);

    int crf = 23;
    render->add_option("--crf", crf)
        ->description("Set the constant rate factor (video only)")
        ->default_val(crf)
        ->check(CLI::Range(0, 51));

    auto build = app.add_subcommand("build", "Build model");

    try {
        app.parse((argc), (argv));
        if (model == "cunet/art" && scale == 4)
            throw std::runtime_error("cunet/art does not support scale factor 4.");
        if (noise == -1 && scale == 1)
            throw std::runtime_error("Noise level -1 does not support scale factor 1.");
    }
    catch (const CLI::ParseError& e) {
        return app.exit(e);
    }
    catch (const std::exception& e) {
        std::cerr << e.what();
        exit(-1);
    };
    // endregion

    const std::vector<std::string> extensions = {
        ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
        ".mp4", ".avi", ".mkv", ".avi"
    };
    auto files = utils::findFilesByExtension(inputPaths, extensions, recursive);

    // region Console callbacks
    trt::MessageCallback messageCallback = [&console](trt::Severity severity, const std::string& message) {
        switch (severity) {
            case trt::Severity::critical:
                console->critical(message);
                break;
            case trt::Severity::error:
                console->error(message);
                break;
            case trt::Severity::warn:
                console->warn(message);
                break;
            case trt::Severity::info:
                console->info(message);
                break;
            case trt::Severity::debug:
                console->debug(message);
                break;
            case trt::Severity::trace:
                console->trace(message);
                break;
        }
    };

    size_t fileIndex = 0;
    size_t fileCount = files.size();
    size_t frameIndex = 0;
    size_t frameCount = 0;
    trt::ProgressCallback progressCallback =
        [&console, &fileIndex, &fileCount, &frameIndex, &frameCount] (int current, int total, double speed) {
        console->info("Rendered file {}/{}, frame {}/{}, batch {}/{} @ {:.2f} it/s",
            fileIndex, fileCount, frameIndex, frameCount, current, total, speed);
    };
    // endregion

    trt::Img2Img engine;
    engine.setMessageCallback(messageCallback);
    engine.setProgressCallback(progressCallback);

    const auto modelPath = "models/" + model + "/"
        + (noise == -1 ? "" : "noise" + std::to_string(noise) + "_")
        + (scale == 1 ? "" : "scale" + std::to_string(scale) + "x")
        + ".onnx";
    std::replace(model.begin(), model.end(), '/', '_');
    const auto suffix = "(" + model + ")"
        + (noise == -1 ? "" : "(noise" + std::to_string(noise) + ")")
        + (scale == 1 ? "" : "(scale" + std::to_string(scale) + ")")
        + (tta ? "(tta)" : "");

    if (render->parsed()) {
        trt::RenderConfig config {
            .deviceId = deviceId,
            .precision = precision,
            .batchSize = batchSize,
            .channels = 3,
            .height = tileSize,
            .width = tileSize,
            .scaling = scale,
            .overlap = cv::Point2d(blend, blend),
            .tta = tta
        };

        if (!engine.load(modelPath, config))
            return -1;
        VideoCapture capture;
        VideoWriter writer;
        cv::Mat inputFrame;
        cv::Mat outputFrame;

        writer.setConstantRateFactor(crf);
        for (auto& file : files) {
            capture.open(file.string());
            inputFrame.create(capture.getFrameSize(), CV_8UC3);
            outputFrame.create(capture.getFrameSize() * scale, CV_8UC3);

            frameIndex = 0;
            frameCount = capture.getFrameCount();

            if (!outputDirectory.empty()) {
                file = outputDirectory / file.filename();
            }
            file.replace_filename(file.stem().string() + suffix);
            if (frameCount == 1) {
                file.replace_extension(".png");
                writer.setFrameRate(1)
                    .setPixelFormat("")
                    .setCodec("");
            } else {
                file.replace_extension(".mp4");
                writer.setFrameRate(capture.getFrameRate())
                    .setPixelFormat(pixelFormat)
                    .setCodec(codec);
            }
            writer.setFrameSize(outputFrame.size());
            writer.setOutputFile(file.string());
            writer.open();

            for (auto i = 0; i < frameCount; i++) {
                capture.read(inputFrame);
                if (!engine.render(inputFrame, outputFrame))
                    return -1;
                writer.write(outputFrame);
                frameIndex++;
            }
            capture.release();
            writer.release();

            fileIndex++;
        }
    } else if (build->parsed()) {
        trt::BuildConfig config {
            .deviceId = deviceId,
            .precision = precision,
            .minBatchSize = batchSize,
            .optBatchSize = batchSize,
            .maxBatchSize = batchSize,
            .minChannels = 3,
            .optChannels = 3,
            .maxChannels = 3,
            .minWidth = tileSize,
            .optWidth = tileSize,
            .maxWidth = tileSize,
            .minHeight = tileSize,
            .optHeight = tileSize,
            .maxHeight = tileSize,
        };
        if (!engine.build(modelPath, config))
            return -1;
    }
}