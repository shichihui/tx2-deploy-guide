# NutSight GPU 版（C++ + TensorRT）

把原来的 ONNX Runtime **CPU** 推理换成 **TensorRT GPU** 推理，目标板：**Jetson Orin Nano (JetPack 5.x / L4T 35.x, TensorRT 8.5, CUDA 11.4)**。
同样一份代码在 x86 + NVIDIA 显卡（CUDA 11/12 + TensorRT 8.x）上也能编译运行，方便在 PC 上先验证。

## 一、与 CPU 版的差别

| 项目 | CPU 版（`D:\Desktop\缺陷`） | 本包（GPU 版） |
|---|---|---|
| 推理后端 | ONNX Runtime CPU EP | **TensorRT**（GPU，FP32/FP16/INT8） |
| 依赖 | onnxruntime + OpenCV | **libnvinfer / libnvonnxparsers / libcudart** + OpenCV |
| 输入 | 512×512 灰度 → 3 通道 | 完全相同 |
| 后处理 | 阈值 + 连通域 + 兜底框 | **完全相同**（一行未改） |
| 输出 | 判定 / 分数 / 热图 / 框 / CSV | 完全相同 |
| 速度（Orin Nano） | TTA×4 ≈ 16.2 s/张，TTA×1 ≈ 4 s/张 | 待板上实测（FP16 预期 **0.3~0.8 s/张**，INT8 更快） |
| 引擎构建 | 无需构建 | **首次需要构建引擎**（Orin Nano 上约 3~10 分钟，1118 个节点；之后直接加载） |

> 分数应与 CPU 版逐位一致（同一份 ONNX，同一套前/后处理）。若 FP16 有轻微差异，属正常（通常 <0.001）。\n>\n> 注意：`model.onnx` 里 TTA×4（0/90/180/270 度）**已经固化进图**，所以一次推理等于 4 次网络前向 —— 这也是它比普通模型慢 4 倍的原因；GPU 上这点开销可以接受，若想更快可再导一份 TTA×1 的模型。

## 二、包里有什么

```
quexian/
├── model.onnx              157 MB  patchcore 模型（screw 类，TTA×4 已固化）
├── model.json              阈值与测试指标（image 0.076539 / pixel 0.090223）
├── samples/                4 张样例（2 OK + 2 NG）
├── calib/                  60 张正常样本（INT8 量化校准用）
├── src/
│   ├── trt_detector.h/.cpp TensorRT 引擎构建 + 推理 + 后处理
│   ├── main.cpp            CLI：单图 / 目录 / 视频
│   ├── imageio.h/.cpp      图片读写（支持中文路径）
│   └── overlay.h/.cpp      热图叠加 + 缺陷框
├── third_party/
│   ├── tensorrt/include/   **TensorRT 8.5 头文件**（板子不用再装 -dev 包）
│   └── cuda/include/       极简 cuda_runtime_api.h（板子没装 CUDA toolkit 时兜底）
├── build.sh                一键编译（自动找 OpenCV / TensorRT / cudart）
├── CMakeLists.txt          也可用 cmake 编译
├── check_env.sh            板子环境自检（先跑这个！）
├── run.sh                  统一入口（--int8 / --fp32 / --image / --dir / --video）
└── 操作指令.txt             一条条照抄的命令清单
```

## 三、第一步：自检环境

```bash
cd ~/quexian
chmod +x *.sh
./check_env.sh
```

重点看三段：

| 输出 | 含义 | 处理 |
|---|---|---|
| `libnvinfer.so.8` 存在 | TensorRT 运行时已装 ✅ | 直接编译 |
| `没有 libnvinfer.so*` | **TensorRT 没装** ❌ | 见第六节 |
| `cuda_runtime_api.h` 缺失但 `libcudart.so*` 存在 | 只缺 CUDA 头文件 | 无所谓，本包自带兜底 |

## 四、编译

```bash
./build.sh
```

脚本会依次检查 g++、OpenCV、TensorRT 头文件、CUDA 头文件、三个运行库，然后编译出 `build/nutsight-cli`。
想用 cmake 也行：`mkdir -p b && cd b && cmake .. && make -j4`。

## 五、运行

```bash
# 1) 默认 FP16，第一次会自动建引擎（Orin Nano 约 3~10 分钟，看到 [build] ... 就是在建）
./run.sh

# 期望输出（与 CPU 版一致）：
#   good_000.png          OK   score=0.0407  热区 0.00%
#   good_001.png          OK   score=0.0455  热区 0.00%
#   scratch_head_000.png  NG   score=0.0852  热区 0.57%   2 个区域
#   thread_side_000.png   OK   score=0.0498  热区 0.00%

# 2) INT8 量化（用 calib/ 里的 60 张正常样本校准）
./run.sh --int8

# 3) 其他用法
./run.sh --image 你的图.png
./run.sh --dir 某个目录 --save-dir out --json out/r.json
./run.sh --video 产线.mp4 --stride 2 --csv out/frames.csv
./run.sh --fp32                       # 全精度（排错用）
./run.sh --int8 --rebuild             # 强制重建引擎
./run.sh --precision int8 --calib-dir 别的校准目录 --rebuild
```

引擎文件是 `model.fp16.engine` / `model.int8.engine`（原地生成，**与设备绑定**，换设备需重建）。

## 六、板上没有 TensorRT 怎么办

`check_env.sh` 显示没有 `libnvinfer.so*`，说明这块板子没装 TensorRT（JetPack 精简安装常见）。
三条路，按可行性排序：

1. **有网**：`sudo apt-get update && sudo apt-get install -y nvidia-tensorrt`
2. **离线**：在 PC 上用 NVIDIA SDK Manager 只勾 `TensorRT` 组件，把下载到的 `*.deb` 拷进板子：
   ```bash
   sudo dpkg -i nv-tensorrt-local-repo-*.deb
   sudo apt-get update && sudo apt-get install -y nvidia-tensorrt
   ```
   需要的是这几个包：`libnvinfer8`、`libnvinfer-plugin8`、`libnvonnxparsers8`（外加 `libnvinfer-bin` 可选，里面有 `trtexec`）。
3. **先退回 CPU 版**：`D:\Desktop\缺陷`（已验证可用，16.2 s/张），等 TensorRT 装好再切 GPU。

> 本包已自带 TensorRT 8.5 的头文件，所以**不需要装 `libnvinfer-dev`**，只要运行时库。

## 七、INT8 量化说明（重要）

- 原理：TensorRT 用 `calib/` 里 60 张**正常样本**做熵校准（`IInt8EntropyCalibrator2`），把卷积/GEMM 的权重与激活量化到 8 bit，从 Tensor Core 拿吞吐。
- 校准缓存会写到 `model.int8.engine.calib`，第二次构建直接复用（想重标就删掉这个文件）。
- **精度必须复核**：INT8 会改变分数分布，原有的 `pixel 0.0902 / image 0.0765` 阈值可能不再适用。
  - 先跑 `./run.sh --int8` 看 4 张样例分数是否还落在原阈值两侧；
  - 若整体偏移，用 `--thr-image`、`--thr-pixel` 临时覆盖，或告诉我，我按新分数重标阈值。
- 一个已实测的结论（本机 x86 CPU 上的 ONNX 动态量化）：把记忆库那块大 GEMM 单独量化**不带来任何加速**（165.0→165.1 MB、1874→1848 ms、分数完全不变）。
  在 **GPU** 上情况不同 —— 卷积主干（WideResNet-50）走 Tensor Core 的 INT8 收益明显，所以这里仍值得试，但要按上面的方法复核精度。

## 八、常见错误

| 报错 | 原因 | 处理 |
|---|---|---|
| `找不到 libnvinfer.so` | TensorRT 未装 | 见第六节 |
| `找不到 libnvonnxparsers.so` | 只装了推理部分，缺 ONNX 解析器 | 装 `libnvonnxparsers8` |
| `deserializeCudaEngine` 失败 | 引擎与板子/版本不匹配（在别的机器上建的） | `./run.sh --rebuild` |
| `ONNX 解析失败` | ONNX 损坏或被改过 | 重新拷 `model.onnx` |
| `CUDA 调用失败（cudaMalloc…）` | 显存不足（INT8 校准时 1~2 GB） | 关掉桌面程序，或减小 `--workspace` |
| 建引擎很慢（>10 分钟） | Orin Nano 建 PatchCore 引擎确实慢 | 耐心等，只需一次；把 `model.*.engine` 备份好 |
| 分数与 CPU 版差很多 | 阈值文件没配对 / INT8 精度偏移 | 先 `--fp32` 复现，再判断 |

## 九、与 Python 端对拍

```bash
# GPU 版
./run.sh --fp32 --dir samples --json out/gpu.json

# 与 CPU 版对比（另一台机器或另一个目录）
#   分数差值应 < 1e-3（FP32），FP16 通常 < 1e-3
```

也可以直接看 4 张样例的分数是否等于 `0.0407 / 0.0455 / 0.0852 / 0.0498`。
