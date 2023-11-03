#include "tensorrt/img2img.h"
#include "videoio/capture.h"
#include "videoio/writer.h"
#include <argparse/argparse.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

#define SPDLOG_LEVEL_NAMES { "TRACE", "DEBUG", "INFO ", "WARN ", "ERROR", "FATAL", "OFF" }
#include <spdlog/sinks/stdout_color_sinks.h>
#include <filesystem>

int main(int argc, char *argv[]) {
    auto console = spdlog::stdout_color_mt("console");
    console->set_level(spdlog::level::info);
    console->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");

    // region Argument Parser
    argparse::ArgumentParser modelParser("model", "", argparse::default_arguments::none);
    modelParser.add_argument("--model")
        .help("Model to use.")
        //.choices("cunet/art", "swin_unet/art", "swin_unet/art_scan", "swin_unet/photo")
        .required();
    modelParser.add_argument("--scale")
        .help("Scale factor to use.")
        //.choices(1, 2, 4)
        .default_value(2)
        .scan<'i', int>();
    modelParser.add_argument("--noise")
        .help("Noise level to use.")
        //.choices(-1, 0, 1, 2, 3)
        .default_value(1)
        .scan<'i', int>();
    modelParser.add_argument("--device")
        .help("GPU Device ID to use.")
        .default_value(0)
        .scan<'i', int>();
    modelParser.add_argument("--precision")
        .help("Precision to use.")
        //.choices("FP16", "TF32")
        .default_value("FP16");
    modelParser.add_argument("--batchSize")
        .help("Model batch size.")
        .default_value(1)
        .scan<'i', int>();
    modelParser.add_argument("--tileSize")
        .help("Model tile size.")
        .nargs(1)
        //.choices(64, 256, 400, 640)
        .default_value(256)
        .scan<'i', int>();

    argparse::ArgumentParser renderCommand("render");
    renderCommand.add_description("Render image(s).");
    renderCommand.add_argument("-i", "--input")
        .help("Input images(s) to render.")
        .nargs(argparse::nargs_pattern::at_least_one)
        .required();
    renderCommand.add_argument("-o", "--output")
        .help("Output images(s) destination.")
        .required();
    renderCommand.add_parents(modelParser);
    renderCommand.add_argument("--blend")
        .help("Percentage of overlap between two tiles to blend.")
        .default_value(1.0 / 16.0)
        .scan<'g', double>();
    renderCommand.add_argument("--tta")
        .help("Use test-time augmentation.")
        .default_value(false)
        .implicit_value(true);

    argparse::ArgumentParser buildCommand("build");
    buildCommand.add_description("Build model.");
    buildCommand.add_parents(modelParser);

    argparse::ArgumentParser program("waifu2x");
    program.add_subparser(renderCommand);
    program.add_subparser(buildCommand);

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cout << err.what() << std::endl;
        std::cout << program;
        exit(-1);
    }
    // endregion

    trt::LogCallback callback = [&console](trt::Severity severity, const std::string& message, const std::string& file, const std::string& function, int line) {
        const auto s = "[" + function + "@" + std::to_string(line) + "] " + message;
        switch (severity) {
            case trt::Severity::critical:
                console->critical(s);
                break;
            case trt::Severity::error:
                console->error(s);
                break;
            case trt::Severity::warn:
                console->warn(s);
                break;
            case trt::Severity::info:
                console->info(s);
                break;
            case trt::Severity::debug:
                console->debug(s);
                break;
            case trt::Severity::trace:
                console->trace(s);
                break;
        }
    };

    trt::Img2Img engine;
    engine.setLogCallback(callback);

    argparse::ArgumentParser* parser;
    if (program.is_subcommand_used(renderCommand)) {
        parser = &renderCommand;
    } else if (program.is_subcommand_used(buildCommand)) {
        parser = &buildCommand;
    } else {
        console->error("No subcommand used.");
        exit(-1);
    }

    // check if model, scale and noise are compatible
    const auto model = parser->get<std::string>("model");
    const auto scale = parser->get<int>("scale");
    const auto noise = parser->get<int>("noise");
    if (model == "cunet/art" && scale == 4) {
        console->error("cunet/art does not support scale factor 4.");
        exit(-1);
    }
    if (noise == -1 && scale == 1) {
        console->error("Noise level -1 does not support scale factor 1.");
        exit(-1);
    }

    const auto modelPath = "models/" + model + "/"
        + (noise == -1 ? "" : "noise" + std::to_string(noise) + "_")
        + (scale == 1 ? "" : "scale" + std::to_string(scale) + "x")
        + ".onnx";

    if (program.is_subcommand_used(renderCommand)) {
        // check if input exists
        const auto inputPaths = renderCommand.get<std::vector<std::string>>("input");
        for (const auto& path : inputPaths) {
            if (!std::filesystem::exists(path)) {
                console->error("Input file \"{}\" does not exist.", path);
                exit(-1);
            }
        }

        // check if output dir exists
        const auto outputPath = renderCommand.get<std::string>("output");
        if (!std::filesystem::exists(outputPath)) {
            console->error("Output directory \"{}\" does not exist.", outputPath);
            exit(-1);
        }

        // check if blend is in range
        const auto blend = renderCommand.get<double>("blend");
        if (blend < 0.0 || blend >= 1.0) {
            console->error("Blend must be in range [0, 1[.");
            exit(-1);
        }

        trt::RenderConfig config {
            .deviceId = parser->get<int>("device"),
            .precision = parser->get<std::string>("precision") == "FP16" ? trt::Precision::FP16 : trt::Precision::TF32,
            .batchSize = parser->get<int>("batchSize"),
            .channels = 3,
            .height = parser->get<int>("tileSize"),
            .width = parser->get<int>("tileSize"),
            .scaling = scale,
            .overlap = cv::Point2d(blend, blend),
            .tta = renderCommand.get<bool>("tta")
        };

        engine.load(modelPath, config);
        VideoCapture capture;
        VideoWriter writer;

        for (const auto& path : inputPaths) {
            capture.open(path);

            const auto frameCount = capture.getFrameCount();
            cv::Mat inputFrame(capture.getFrameSize(), CV_8UC3);
            cv::Mat outputFrame(capture.getFrameSize() * scale, CV_8UC3);

            writer.setFrameSize(outputFrame.size());

            if (frameCount == 1) {
                writer.setOutputFile(outputPath + "\\out.png");
            } else {
                writer.setOutputFile(outputPath + "\\out.mp4")
                    .setFrameRate(capture.getFrameRate())
                    .setPixelFormat("yuv420p")
                    .setCodec("libx264");
            }
            writer.open();

            for (auto i = 0; i < frameCount; i++) {
                capture.read(inputFrame);
                engine.render(inputFrame, outputFrame);
                writer.write(outputFrame);
            }
            capture.release();
            writer.release();
        }
    } else if (program.is_subcommand_used(buildCommand)) {

    }
}