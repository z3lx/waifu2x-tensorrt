cmake_minimum_required(VERSION 3.26)
project(waifu2x_tensorrt)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(waifu2x_tensorrt src/main.cpp)
target_include_directories(waifu2x_tensorrt PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(waifu2x_tensorrt PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES})