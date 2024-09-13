# waifu2x-tensorrt
waifu2x-tensorrt is a TensorRT implementation of the waifu2x super-resolution model. This project aims to improve the inference speed for faster image upscaling on NVIDIA GPUs. Supports images and videos alike.

**Note**: This project is currently under active development by a high school student (that's me!). As a result, some features may be missing, and bugs can be expected. Contributions and feedback are welcome though!

## Installation
### Precompiled Version
1. Download the precompiled binary:
   - Visit the [Releases](https://github.com/z3lx/waifu2x-tensorrt/releases) section of this repository
   - Download the precompiled binary for your operating system (only win64 currently)
2. Download Models:
   - In the same [Releases](https://github.com/z3lx/waifu2x-tensorrt/releases) section, download the models archive.
3. Extract Files:
   - Extract the precompiled binary archive to a location of your choice.
   - Extract the models archive within the same folder as the precompiled binary.

### Building from source
1. Install the dependencies:
   - [CMake](https://cmake.org/)
   - [OpenCV](https://opencv.org/releases/) with CUDA support (version >= 4.8.0)
   - [TensorRT](https://developer.nvidia.com/tensorrt)
   - [CUDA toolkit](https://developer.nvidia.com/cuda-downloads)
   - [cuDNN](https://developer.nvidia.com/cudnn)
   
   Refer to the [support matrix](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html) for compatible TensorRT, CUDA, and cuDNN versions.
3. Clone the repository:
```
git clone https://github.com/z3lx/waifu2x-tensorrt.git
cd waifu2x-tensorrt
```
3. Build the project files using CMake:
```
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Usage
```
waifu2x-tensorrt
Usage: waifu2x-tensorrt [OPTIONS] SUBCOMMAND

Options:
  -h,--help                                                     Print this help message and exit
  --model TEXT:{cunet/art,swin_unet/art,swin_unet/art_scan,swin_unet/photo} REQUIRED
                                                                Set the model to use
  --scale INT:{1,2,4} REQUIRED                                  Set the scale factor
  --noise INT:{-1,0,1,2,3} REQUIRED                             Set the noise level
  --batchSize INT:POSITIVE REQUIRED                             Set the batch size
  --tileSize INT:{64,256,400,640} REQUIRED                      Set the tile size
  --device INT:NONNEGATIVE [0]                                  Set the GPU device ID
  --precision ENUM:value in {fp16->1,tf32->0} OR {1,0} [1]      Set the precision

Subcommands:
render
  Render image(s)/video(s)
  Options:
    -i,--input TEXT:PATH(existing) ... REQUIRED                   Set the input paths
    --recursive                                                   Search for input files recursively
    -o,--output TEXT:DIR                                          Set the output directory
    --blend FLOAT:{0.125,0.0625,0.03125,0} [0.0625]               Set the percentage of overlap between two tiles to blend
    --tta [0]                                                     Enable test-time augmentation
    --codec TEXT [libx264]                                        Set the codec (video only)
    --pix_fmt TEXT [yuv420p]                                      Set the pixel format (video only)
    --crf INT:INT in [0 - 51] [23]                                Set the constant rate factor (video only)

build
  Build model
```

### Building a model
Before being able to upscale an image or a video using a particular configuration, the model needs to be built into an optimized engine. To do so, use the build subcommand and specify the model, the scale, the noise, the batch size, and the tile size: 
```
waifu2x-tensorrt build --model swin_unet/art --scale 4 --noise 3 --batchSize 4 --tileSize 256
```
Depending on the configuration, this process might take a couple of minutes to complete, and TensorRT might fail if VRAM is insufficient. 

### Upscaling an image/video
To upscale an image and/or a video, use the render subcommand and specify the upscaling configuration and input files:
```
waifu2x-tensorrt render --model swin_unet/art --scale 4 --noise 3 --batchSize 4 --tileSize 256 --input path/to/file1.png path/to/file2.mp4 path/to/files
```

## Contributing
Contributions are welcome! If you decide to tackle any of these tasks or have your own ideas for improvement, please create an issue to discuss changes before submitting a pull request.
### TODO
- Add alpha support
- Add wide char support
- Add more options for video
- Further optimize render
- Add GUI
### Known bugs
- Program might not work correctly on linux/mac
- Upscaling does not tile correctly on some models when batchSize > 1

## Acknowledgments
- [waifu2x](https://github.com/nagadomi/nunif/tree/master/waifu2x): The original waifu2x super-resolution model
- [tensorrt-cpp-api](https://github.com/cyrusbehr/tensorrt-cpp-api): A reference for TensorRT integration in C++

## License
This project is licensed under the [MIT License](https://github.com/z3lx/waifu2x-tensorrt/blob/main/LICENSE).
