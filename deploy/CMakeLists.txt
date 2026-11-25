cmake_minimum_required(VERSION 3.10)
project(nutsight_serving_gpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)

# ---- TensorRT 头文件 ----
set(TRT_INCLUDE_HINTS
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorrt/include
    /usr/include/aarch64-linux-gnu
    /usr/include)
find_path(NVINFER_INCLUDE_DIR NvInfer.h HINTS ${TRT_INCLUDE_HINTS})

# ---- CUDA 头文件（系统有就用系统的，否则用本包自带的极简声明） ----
find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h
          HINTS /usr/local/cuda/include /usr/local/cuda-11.4/include
                ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cuda/include)

# ---- 运行库 ----
find_library(NVINFER_LIB nvinfer HINTS /usr/lib/aarch64-linux-gnu ${TRT_INCLUDE_HINTS})
find_library(NVONNXPARSERS_LIB nvonnxparsers HINTS /usr/lib/aarch64-linux-gnu)
find_library(CUDART_LIB cudart HINTS /usr/local/cuda/lib64 /usr/lib/aarch64-linux-gnu)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin HINTS /usr/lib/aarch64-linux-gnu)

if(NOT NVINFER_INCLUDE_DIR OR NOT NVINFER_LIB OR NOT NVONNXPARSERS_LIB OR NOT CUDART_LIB)
  message(FATAL_ERROR
    "找不到 TensorRT 或 CUDA 运行时。\n"
    "  NvInfer.h        : ${NVINFER_INCLUDE_DIR}\n"
    "  libnvinfer       : ${NVINFER_LIB}\n"
    "  libnvonnxparsers : ${NVONNXPARSERS_LIB}\n"
    "  libcudart        : ${CUDART_LIB}\n"
    "请先跑 ./check_env.sh 看看缺什么，或直接用 ./build.sh 手工编译。")
endif()

add_executable(nutsight-cli
    src/main.cpp
    src/trt_detector.cpp
    src/imageio.cpp
    src/overlay.cpp)

target_include_directories(nutsight-cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${NVINFER_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS})

target_link_libraries(nutsight-cli PRIVATE
    ${NVINFER_LIB} ${NVONNXPARSERS_LIB} ${CUDART_LIB} ${OpenCV_LIBS})

if(NVINFER_PLUGIN_LIB)
  target_link_libraries(nutsight-cli PRIVATE ${NVINFER_PLUGIN_LIB})
endif()

set_target_properties(nutsight-cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
