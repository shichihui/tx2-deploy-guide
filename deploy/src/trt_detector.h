// TensorRT 版 PatchCore 检测器（GPU 推理）
//
// 与 ORT CPU 版（detector.h）保持完全一致的对外语义：
//   - 预处理：BGR → GRAY(BGR2GRAY) → resize(INTER_AREA, size×size) → /255 → 三通道堆叠 → NCHW
//   - 后处理：图像分数 vs 图像阈值；像素阈值兜底（NG 用 min(thr_pixel, 0.75*map_max)）；
//             连通域 8 邻域、面积 ≥12、按面积降序取前 8；NG 但无连通域时在峰值画固定框
//
// 引擎（.engine）由 ONNX 在本机构建：同一台设备上 FP16 通常与 FP32 分数几乎一致，
// INT8 需要校准集（本项目用 60 张正常样本，熵校准）。
#pragma once

#include <cuda_runtime_api.h>

#include <NvInfer.h>

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace nutsight {

struct Region {
  int x{0};
  int y{0};
  int w{0};
  int h{0};
  int area{0};
};

struct Prediction {
  float score{0.f};
  bool is_defect{false};
  float hot_ratio{0.f};
  std::vector<Region> regions;
  double elapsed_ms{0.0};
  cv::Mat anomaly_map;  // CV_32F，模型输出尺寸（screw 为 32×32 网格）
};

enum class Precision { kFP32, kFP16, kINT8 };

Precision parsePrecision(const std::string& s);
const char* precisionName(Precision p);

struct TrtOptions {
  std::string onnx_path;     // 构建引擎用（--precision 决定 FP32/FP16/INT8）
  std::string engine_path;   // 已构建的引擎；不存在则自动构建
  std::string calib_dir;     // INT8 校准图目录（第一次构建 INT8 时必需）
  std::string calib_cache;   // INT8 校准缓存（默认 <engine>.calib，可复用）
  Precision precision{Precision::kFP16};
  int image_size{512};
  int workspace_mb{1024};
  int device{0};
  bool verbose{false};
  bool rebuild{false};  // 已有引擎也强制重建
};

class TrtDetector {
 public:
  explicit TrtDetector(const TrtOptions& opt);
  ~TrtDetector();

  TrtDetector(const TrtDetector&) = delete;
  TrtDetector& operator=(const TrtDetector&) = delete;

  Prediction predict(const cv::Mat& bgr, float thr_image, float thr_pixel);

  int imageSize() const { return size_; }
  double loadMs() const { return load_ms_; }
  const std::string& enginePath() const { return engine_path_; }
  const char* precision() const { return precisionName(opt_.precision); }
  bool builtNow() const { return built_now_; }
  int64_t mapWidth() const { return map_w_; }
  int64_t mapHeight() const { return map_h_; }

 private:
  void buildEngine();
  void loadEngine();
  void setupBindings();

  TrtOptions opt_;
  int size_{512};
  double load_ms_{0.0};
  bool built_now_{false};
  std::string engine_path_;

  std::shared_ptr<nvinfer1::ILogger> logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  std::vector<std::string> binding_names_;
  int in_idx_{-1};
  int in_size_{0};
  int out_score_idx_{-1};
  int out_map_idx_{-1};
  int64_t map_w_{0};
  int64_t map_h_{0};

  void* dev_in_{nullptr};
  void* dev_score_{nullptr};
  void* dev_map_{nullptr};
  cudaStream_t stream_{nullptr};
  std::vector<float> host_in_;
  std::vector<float> host_score_;
  std::vector<float> host_map_;
};

}  // namespace nutsight
