cmake_minimum_required(VERSION 3.26)
project(waifu2x_tensorrt)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# Download plog
include(FetchContent)
FetchContent_Declare(
        plog
        GIT_REPOSITORY https://github.com/SergiusTheBest/plog.git
)
FetchContent_GetProperties(plog)
if(NOT plog_POPULATED)
    FetchContent_Populate(plog)
endif()
add_subdirectory(${plog_SOURCE_DIR})

add_executable(waifu2x_tensorrt
        src/main.cpp
        src/tensorrt/engine.cpp
        src/tensorrt/engine.h
        src/tensorrt/config.h
        src/tensorrt/logger.cpp
        src/tensorrt/logger.h
        src/plog/plog.h
)

target_include_directories(waifu2x_tensorrt PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        ${plog_SOURCE_DIR}/include
)

target_link_libraries(waifu2x_tensorrt PUBLIC
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${TensorRT_LIBRARIES}
)