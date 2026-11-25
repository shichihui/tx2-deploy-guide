#!/usr/bin/env bash
# =====================================================================
# NutSight GPU 版（TensorRT）—— 一键编译
# 目标：Jetson Orin Nano (JetPack 5.x / L4T 35.x, TensorRT 8.5, CUDA 11.4)
# 依赖：g++ >= 9、OpenCV4 开发包、TensorRT 运行时(libnvinfer/libnvonnxparsers)、libcudart
# =====================================================================
set -u

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here" || { echo "[错误] 进不去目录 $here"; exit 1; }

red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
yellow(){ printf '\033[33m%s\033[0m\n' "$*"; }

echo "===== 1. 编译器 ====="
if ! command -v g++ >/dev/null 2>&1; then
  red "[错误] 找不到 g++。离线安装：cd debs && sudo dpkg -i ./*.deb"
  exit 1
fi
g++ --version | head -1

echo
echo "===== 2. OpenCV ====="
if pkg-config --exists opencv4 2>/dev/null; then
  echo "OpenCV $(pkg-config --modversion opencv4)"
  cv_cflags="$(pkg-config --cflags opencv4)"
  cv_libs="$(pkg-config --libs opencv4)"
elif [ -f /usr/include/opencv4/opencv2/core.hpp ]; then
  yellow "[提示] 没有 opencv4.pc，但头文件存在，用固定参数编译"
  cv_cflags="-I/usr/include/opencv4"
  cv_libs="-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio"
else
  red "[错误] 找不到 OpenCV4 开发包。请看 README 的「装 OpenCV」一节"
  exit 1
fi

echo
echo "===== 3. TensorRT 头文件 ====="
trt_inc=""
for d in "$here/third_party/tensorrt/include" /usr/include/aarch64-linux-gnu /usr/include; do
  if [ -f "$d/NvInfer.h" ]; then trt_inc="$d"; break; fi
done
if [ -z "$trt_inc" ]; then
  red "[错误] 找不到 NvInfer.h"
  exit 1
fi
echo "头文件：$trt_inc"

echo
echo "===== 4. CUDA 头文件 ====="
cuda_inc=""
for d in /usr/local/cuda/include /usr/local/cuda-11.4/include "$here/third_party/cuda/include"; do
  if [ -f "$d/cuda_runtime_api.h" ]; then cuda_inc="$d"; break; fi
done
if [ -z "$cuda_inc" ]; then
  red "[错误] 找不到 cuda_runtime_api.h"
  exit 1
fi
echo "头文件：$cuda_inc"

echo
echo "===== 5. 定位运行库 ====="
find_lib() { # $1 = 库名前缀，返回文件的完整路径（优先不带版本号的 .so）
  local name="$1" p
  for p in /usr/lib/aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/tegra /usr/lib /usr/local/cuda/lib64 \
           /usr/local/cuda-11.4/lib64 /usr/local/cuda/targets/aarch64-linux/lib; do
    [ -d "$p" ] || continue
    if [ -f "$p/lib${name}.so" ]; then echo "$p/lib${name}.so"; return 0; fi
  done
  for p in /usr/lib/aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/tegra /usr/lib /usr/local/cuda/lib64 \
           /usr/local/cuda-11.4/lib64 /usr/local/cuda/targets/aarch64-linux/lib; do
    [ -d "$p" ] || continue
    local f
    f="$(ls -1 "$p"/lib${name}.so.* 2>/dev/null | sort -V | tail -1)"
    if [ -n "$f" ]; then echo "$f"; return 0; fi
  done
  return 1
}

trt_lib="$(find_lib nvinfer)"          || { red "[错误] 找不到 libnvinfer.so（TensorRT 运行时未安装）"; exit 1; }
onnx_lib="$(find_lib nvonnxparsers)"   || { red "[错误] 找不到 libnvonnxparsers.so（TensorRT ONNX 解析器未安装）"; exit 1; }
cudart_lib="$(find_lib cudart)"        || { red "[错误] 找不到 libcudart.so（CUDA 运行时未安装）"; exit 1; }
plugin_lib="$(find_lib nvinfer_plugin)" || true

echo "libnvinfer      : $trt_lib"
echo "libnvonnxparsers: $onnx_lib"
echo "libcudart       : $cudart_lib"
if [ -n "$plugin_lib" ]; then echo "libnvinfer_plugin: $plugin_lib"; fi

# 把（可能带版本号的）库文件软链到 build/lib-deps，用 -l 的方式让链接器自己处理符号链接
mkdir -p build/lib-deps
ln -sf "$trt_lib"    build/lib-deps/libnvinfer.so
ln -sf "$onnx_lib"   build/lib-deps/libnvonnxparsers.so
ln -sf "$cudart_lib" build/lib-deps/libcudart.so
if [ -n "$plugin_lib" ]; then ln -sf "$plugin_lib" build/lib-deps/libnvinfer_plugin.so; fi

echo
echo "===== 6. 编译 ====="
lib_dirs="-Lbuild/lib-deps -L$(dirname "$trt_lib") -L$(dirname "$cudart_lib")"
ln_flags="-lnvinfer -lnvonnxparsers -lcudart"
if [ -n "$plugin_lib" ]; then ln_flags="$ln_flags -lnvinfer_plugin"; fi

set -x
g++ -std=c++17 -O3 -pthread -Wno-deprecated-declarations \
    -o build/nutsight-cli \
    src/main.cpp src/trt_detector.cpp src/imageio.cpp src/overlay.cpp \
    -Isrc -I"$trt_inc" -I"$cuda_inc" \
    $cv_cflags \
    $lib_dirs $ln_flags \
    $cv_libs \
    -Wl,-rpath,"$(dirname "$trt_lib"):$(dirname "$cudart_lib"):\$ORIGIN/lib-deps"
rc=$?
set +x

if [ $rc -ne 0 ]; then
  red "[错误] 编译失败（把上面最后几行报错发我）"
  exit $rc
fi

green "[完成] 可执行文件：$here/build/nutsight-cli"
./build/nutsight-cli --help | head -4
echo
echo "下一步："
echo "  ./build/nutsight-cli --image samples/scratch_head_000.png            # 首次会用 ONNX 建 FP16 引擎（约 1~3 分钟）"
echo "  ./build/nutsight-cli --precision int8 --calib-dir calib --image samples/scratch_head_000.png   # INT8 量化版"
echo "  ./build/nutsight-cli --dir samples --save-dir out"
