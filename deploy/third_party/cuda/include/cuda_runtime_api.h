// 极简 cuda_runtime_api.h —— 只为本项目用到的 CUDA Runtime 接口提供声明。
//
// 目的：JetPack 的 L4T CUDA 运行时（libcudart.so）通常随系统安装，但
// CUDA 头文件只有装了 cuda-toolkit 才有。TensorRT 头文件只需要
// cudaStream_t / cudaEvent_t 两个类型，本项目另需 cudaMalloc / cudaMemcpy /
// cudaStream* 等少数函数，因此这里给出与官方 ABI 等价的最小声明。
//
// 若板子上确实有官方 CUDA 头文件（/usr/local/cuda/include），build.sh 会
// 优先用系统的，不会用到本文件。
#ifndef NUTSIGHT_CUDA_RUNTIME_API_SHIM_H
#define NUTSIGHT_CUDA_RUNTIME_API_SHIM_H

#include <stddef.h>

typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;

enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4
};

#ifdef __cplusplus
extern "C" {
#endif

int cudaMalloc(void** devPtr, size_t size);
int cudaFree(void* devPtr);
int cudaMemset(void* devPtr, int value, size_t count);
int cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
int cudaMemcpyAsync(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind,
                    cudaStream_t stream);
int cudaStreamCreate(cudaStream_t* pStream);
int cudaStreamDestroy(cudaStream_t stream);
int cudaStreamSynchronize(cudaStream_t stream);
int cudaEventCreate(cudaEvent_t* event);
int cudaEventDestroy(cudaEvent_t event);
int cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
int cudaEventSynchronize(cudaEvent_t event);
int cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);
int cudaGetDeviceCount(int* count);
int cudaSetDevice(int device);
int cudaGetDevice(int* device);
int cudaDeviceSynchronize(void);
const char* cudaGetErrorString(int err);
int cudaGetLastError(void);

#ifdef __cplusplus
}
#endif

#endif  // NUTSIGHT_CUDA_RUNTIME_API_SHIM_H
