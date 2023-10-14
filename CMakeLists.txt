cmake_minimum_required(VERSION 3.26)
project(waifu2x_tensorrt)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# spdlog
include(FetchContent)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.12.0
)
FetchContent_GetProperties(spdlog)
if(NOT spdlog_POPULATED)
    FetchContent_Populate(spdlog)
endif()
add_subdirectory(${spdlog_SOURCE_DIR})

# nlohmann/json
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
FetchContent_GetProperties(json)
if (NOT json_POPULATED)
    FetchContent_Populate(json)
endif()
add_subdirectory(${json_SOURCE_DIR})

add_executable(waifu2x_tensorrt
    src/main.cpp
    src/tensorrt/config.h
    src/tensorrt/helper.h
    src/tensorrt/img2img.h
    src/tensorrt/img2img_base.cpp
    src/tensorrt/img2img_build.cpp
    src/tensorrt/img2img_infer.cpp
    src/tensorrt/img2img_load.cpp
    src/tensorrt/img2img_render.cpp
    src/tensorrt/img2img_serialize.cpp
    src/tensorrt/logger.cpp
    src/tensorrt/logger.h
    src/utilities/sha256.h
    src/utilities/time.h
    src/utilities/path.h
)

target_include_directories(waifu2x_tensorrt PUBLIC
    ${PROJECT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    ${spdlog_SOURCE_DIR}/include
    ${json_SOURCE_DIR}/include
)

target_link_libraries(waifu2x_tensorrt PUBLIC
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TensorRT_LIBRARIES}
)