# waifu2x-tensorrt
waifu2x-tensorrt is a TensorRT implementation of the original waifu2x super-resolution model, found [here](https://github.com/nagadomi/nunif/tree/master/waifu2x). This project aims to improve the inference speed of the upscaling process on NVIDIA GPUs.

**Note**: This project is currently under active development by a high school student (that's me!). As a result, some features may be missing, and bugs can be expected. Contributions and feedback are welcome though!

## Installation
### Precompiled Version
A precompiled version of the program can be found in the [Releases](https://github.com/z3lx/waifu2x-tensorrt/releases) section.

### Building from source
1. Install the dependencies:
   - [CMake](https://cmake.org/)
   - [OpenCV](https://opencv.org/releases/) with CUDA support (version >= 4.8.0)
   - [TensorRT](https://developer.nvidia.com/tensorrt)
   - [CUDA toolkit](https://developer.nvidia.com/cuda-downloads)
   
   Refer to the [support matrix](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html) for compatible CUDA and TensorRT versions.
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
Depending on the configuration, this process might take a while to complete, and TensorRT might fail if VRAM is insufficient. 

### Upscaling an image or a video
To upscale an image or a video, use the render subcommand and specify the upscaling configuration and input files:
```
waifu2x-tensorrt render --model swin_unet/art --scale 4 --noise 3 --batchSize 4 --tileSize 256 --input path/to/file1.png path/to/file2.mp4 path/to/files
```

## Contributing
Contributions are welcome! If you decide to tackle any of these tasks or have your own ideas for improvement, please create an issue to discuss changes before submitting a pull request.
### TODO
- Add alpha support
- Add wide char support
- Add more options for video
- Add GUI

## License
This project is licensed under the MIT License.
