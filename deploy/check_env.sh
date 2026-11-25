#!/usr/bin/env bash
# 板子环境自检：把输出整段发给开发者，就能判断缺什么。
echo "===== 系统 ====="
cat /etc/nv_tegra_release 2>/dev/null | head -2
. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME"
uname -m
echo
echo "===== 磁盘 ====="
df -h / | tail -1
echo
echo "===== 编译器 / 构建工具 ====="
for c in g++ gcc cc make cmake pkg-config nvcc; do
  if command -v $c >/dev/null 2>&1; then printf '%-12s %s\n' "$c" "$($c --version 2>&1 | head -1)"; else printf '%-12s 缺失\n' "$c"; fi
done
echo
echo "===== OpenCV ====="
pkg-config --modversion opencv4 2>/dev/null || echo "opencv4.pc 缺失"
ls -d /usr/include/opencv4 2>/dev/null || echo "/usr/include/opencv4 缺失"
echo
echo "===== CUDA ====="
ls /usr/local/cuda*/bin/nvcc 2>/dev/null || echo "没有 /usr/local/cuda*/bin/nvcc"
ls /usr/local/cuda*/include/cuda_runtime_api.h 2>/dev/null || echo "没有 cuda 头文件"
ls -l /usr/lib/aarch64-linux-gnu/libcudart.so* 2>/dev/null || echo "没有 libcudart.so*"
echo
echo "===== TensorRT ====="
ls -l /usr/lib/aarch64-linux-gnu/libnvinfer*.so* 2>/dev/null | head -8 || echo "没有 libnvinfer.so*"
ls -l /usr/lib/aarch64-linux-gnu/libnvonnxparsers.so* 2>/dev/null | head -4 || echo "没有 libnvonnxparsers.so*"
ls /usr/include/aarch64-linux-gnu/NvInfer.h /usr/include/NvInfer.h 2>/dev/null || echo "没有 NvInfer.h（本包已自带，不算问题）"
command -v trtexec >/dev/null 2>&1 && trtexec --version 2>&1 | head -3 || echo "没有 trtexec"
ls /usr/src/tensorrt/bin/trtexec 2>/dev/null || true
echo
echo "===== GPU ====="
ls /dev/nvhost-gpu /dev/nvgpu 2>/dev/null || echo "未发现 GPU 设备节点"
tegrastats --interval 1000 2>/dev/null | head -2 || echo "没有 tegrastats"
echo
echo "===== 内存 ====="
free -h | head -2
