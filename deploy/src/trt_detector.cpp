#include "trt_detector.h"

#include <NvOnnxParser.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace nutsight {
namespace fs = std::filesystem;

namespace {

double nowMs() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

void throwCuda(int code, const char* what) {
  if (code != 0) {
    throw std::runtime_error(std::string("CUDA 调用失败（") + what + "），错误码 " +
                             std::to_string(code) + "：" +
                             (cudaGetErrorString(code) ? cudaGetErrorString(code) : "unknown"));
  }
}

class TrtLogger : public nvinfer1::ILogger {
 public:
  explicit TrtLogger(bool verbose) : verbose_(verbose) {}
  void log(Severity sev, const char* msg) noexcept override {
    if (verbose_ || sev <= Severity::kWARNING) {
      std::fprintf(stderr, "[TensorRT] %s\n", msg);
    }
  }

 private:
  bool verbose_{false};
};

/// INT8 熵校准：逐张喂入正常样本（预处理与推理完全一致）。
class Int8Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
 public:
  Int8Calibrator(const std::string& dir, int size, const std::string& cache_file)
      : size_(size), cache_file_(cache_file) {
    std::error_code ec;
    for (const auto& e : fs::recursive_directory_iterator(dir, ec)) {
      if (!e.is_regular_file()) continue;
      const std::string ext = e.path().extension().string();
      std::string low;
      for (char c : ext) low.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
      if (low == ".png" || low == ".jpg" || low == ".jpeg" || low == ".bmp") {
        files_.push_back(e.path().string());
      }
    }
    std::sort(files_.begin(), files_.end());
    if (files_.empty()) {
      throw std::runtime_error("INT8 校准目录里没有图片：" + dir);
    }
    host_.resize(static_cast<size_t>(3) * size_ * size_);
    throwCuda(cudaMalloc(&device_, host_.size() * sizeof(float)), "cudaMalloc(calib)");
    throwCuda(cudaStreamCreate(&stream_), "cudaStreamCreate(calib)");
  }

  ~Int8Calibrator() override {
    if (device_) cudaFree(device_);
    if (stream_) cudaStreamDestroy(stream_);
  }

  int getBatchSize() const noexcept override { return 1; }

  bool getBatch(void* bindings[], char const* /*names*/[], int32_t nb_bindings) noexcept override {
    if (nb_bindings < 1) return false;
    if (cursor_ >= files_.size()) return false;

    const std::string& path = files_[cursor_++];
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) {
      std::fprintf(stderr, "[calib] 跳过无法读取的图片：%s\n", path.c_str());
      return getBatch(bindings, nullptr, nb_bindings);
    }
    fillTensor(img, host_.data());

    if (cudaMemcpy(device_, host_.data(), host_.size() * sizeof(float), cudaMemcpyHostToDevice) != 0) {
      return false;
    }
    bindings[0] = device_;
    return true;
  }

  const void* readCalibrationCache(size_t& length) noexcept override {
    std::ifstream fin(cache_file_, std::ios::binary);
    if (!fin) {
      length = 0;
      return nullptr;
    }
    cache_.assign(std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>());
    length = cache_.size();
    std::fprintf(stderr, "[calib] 复用校准缓存：%s（%zu 字节）\n", cache_file_.c_str(), cache_.size());
    return cache_.empty() ? nullptr : cache_.data();
  }

  void writeCalibrationCache(const void* cache, size_t length) noexcept override {
    std::ofstream fout(cache_file_, std::ios::binary);
    if (!fout) return;
    fout.write(static_cast<const char*>(cache), static_cast<std::streamsize>(length));
    std::fprintf(stderr, "[calib] 已写入校准缓存：%s（%zu 字节）\n", cache_file_.c_str(), length);
  }

 private:
  void fillTensor(const cv::Mat& bgr, float* dst) const {
    cv::Mat gray;
    if (bgr.channels() == 3) {
      cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    } else {
      gray = bgr;
    }
    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(size_, size_), 0, 0, cv::INTER_AREA);
    const size_t plane = static_cast<size_t>(size_) * size_;
    for (int y = 0; y < size_; ++y) {
      const uint8_t* row = resized.ptr<uint8_t>(y);
      for (int x = 0; x < size_; ++x) {
        const float v = static_cast<float>(row[x]) / 255.0f;
        const size_t idx = static_cast<size_t>(y) * size_ + x;
        dst[idx] = v;
        dst[plane + idx] = v;
        dst[2 * plane + idx] = v;
      }
    }
  }

  std::vector<std::string> files_;
  size_t cursor_{0};
  int size_{512};
  std::string cache_file_;
  std::vector<char> cache_;
  std::vector<float> host_;
  void* device_{nullptr};
  cudaStream_t stream_{nullptr};
};

int64_t volume(const nvinfer1::Dims& d) {
  if (d.nbDims < 1) return 1;
  int64_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) {
    if (d.d[i] < 0) throw std::runtime_error("引擎维度含动态轴，本项目不支持动态 shape");
    v *= d.d[i];
  }
  return v;
}

}  // namespace

Precision parsePrecision(const std::string& s) {
  std::string t;
  for (char c : s) t.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  if (t == "fp32" || t == "32" || t == "float32") return Precision::kFP32;
  if (t == "int8" || t == "8") return Precision::kINT8;
  return Precision::kFP16;
}

const char* precisionName(Precision p) {
  switch (p) {
    case Precision::kFP32: return "fp32";
    case Precision::kINT8: return "int8";
    default: return "fp16";
  }
}

TrtDetector::TrtDetector(const TrtOptions& opt) : opt_(opt), size_(opt.image_size) {
  const double t0 = nowMs();
  throwCuda(cudaSetDevice(opt_.device), "cudaSetDevice");
  logger_ = std::make_shared<TrtLogger>(opt_.verbose);

  engine_path_ = opt_.engine_path;
  if (engine_path_.empty()) {
    engine_path_ = opt_.onnx_path;
    const size_t dot = engine_path_.find_last_of('.');
    if (dot != std::string::npos) engine_path_ = engine_path_.substr(0, dot);
    engine_path_ += std::string(".") + precisionName(opt_.precision) + ".engine";
  }

  const bool exists = fs::exists(engine_path_);
  if (!exists || opt_.rebuild) {
    if (opt_.onnx_path.empty()) {
      throw std::runtime_error("引擎不存在且没有提供 --onnx，无法构建：" + engine_path_);
    }
    buildEngine();
    built_now_ = true;
  }
  loadEngine();
  setupBindings();
  load_ms_ = nowMs() - t0;
}

TrtDetector::~TrtDetector() {
  if (dev_in_) cudaFree(dev_in_);
  if (dev_score_) cudaFree(dev_score_);
  if (dev_map_) cudaFree(dev_map_);
  if (stream_) cudaStreamDestroy(stream_);
}

void TrtDetector::buildEngine() {
  std::ifstream fin(opt_.onnx_path, std::ios::binary);
  if (!fin) throw std::runtime_error("打不开 ONNX：" + opt_.onnx_path);
  std::vector<char> onnx_data((std::istreambuf_iterator<char>(fin)),
                              std::istreambuf_iterator<char>());
  fin.close();

  std::fprintf(stderr, "[build] 用 %s 构建 %s 引擎（%zu 字节 ONNX）…\n", opt_.onnx_path.c_str(),
               precisionName(opt_.precision), onnx_data.size());

  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(*logger_));
  if (!builder) throw std::runtime_error("创建 TensorRT IBuilder 失败（libnvinfer 不可用？）");

  const uint32_t flags = 1U << static_cast<uint32_t>(
      nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flags));

  std::unique_ptr<nvonnxparser::IParser> parser(
      nvonnxparser::createParser(*network, *logger_));
  if (!parser) throw std::runtime_error("创建 ONNX 解析器失败（libnvonnxparsers 不可用？）");

  if (!parser->parse(onnx_data.data(), static_cast<int>(onnx_data.size()))) {
    for (int i = 0; i < parser->getNbErrors(); ++i) {
      std::fprintf(stderr, "[build] ONNX 解析错误：%s\n", parser->getError(i)->desc());
    }
    throw std::runtime_error("ONNX 解析失败");
  }

  std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  const size_t ws = static_cast<size_t>(std::max(64, opt_.workspace_mb)) * 1024ull * 1024ull;
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, ws);

  std::unique_ptr<Int8Calibrator> calibrator;
  if (opt_.precision == Precision::kINT8) {
    if (!builder->platformHasFastInt8()) {
      std::fprintf(stderr, "[build] 警告：该设备未报告 INT8 快速路径，仍尝试构建\n");
    }
    if (opt_.calib_dir.empty()) {
      throw std::runtime_error("INT8 构建需要 --calib-dir（正常样本图片目录）");
    }
    std::string cache = opt_.calib_cache;
    if (cache.empty()) cache = engine_path_ + ".calib";
    calibrator = std::make_unique<Int8Calibrator>(opt_.calib_dir, size_, cache);
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    config->setInt8Calibrator(calibrator.get());
  } else if (opt_.precision == Precision::kFP16) {
    if (!builder->platformHasFastFp16()) {
      std::fprintf(stderr, "[build] 警告：该设备未报告 FP16 快速路径，仍尝试构建\n");
    }
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }

  std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  if (!plan || plan->size() == 0) throw std::runtime_error("构建引擎失败（序列化输出为空）");

  const std::string tmp = engine_path_ + ".tmp";
  {
    std::ofstream fout(tmp, std::ios::binary);
    if (!fout) throw std::runtime_error("无法写引擎文件：" + tmp);
    fout.write(static_cast<const char*>(plan->data()), static_cast<std::streamsize>(plan->size()));
  }
  std::error_code ec;
  fs::rename(tmp, engine_path_, ec);
  if (ec) {
    fs::remove(engine_path_, ec);
    fs::rename(tmp, engine_path_, ec);
  }
  std::fprintf(stderr, "[build] 引擎已保存：%s（%.1f MB）\n", engine_path_.c_str(),
               static_cast<double>(plan->size()) / (1024.0 * 1024.0));
}

void TrtDetector::loadEngine() {
  std::ifstream fin(engine_path_, std::ios::binary);
  if (!fin) throw std::runtime_error("打不开引擎文件：" + engine_path_);
  std::vector<char> data((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
  fin.close();

  runtime_.reset(nvinfer1::createInferRuntime(*logger_));
  if (!runtime_) throw std::runtime_error("创建 TensorRT IRuntime 失败");

  engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
  if (!engine_) throw std::runtime_error("反序列化引擎失败（引擎与 TensorRT 版本不匹配？）");

  context_.reset(engine_->createExecutionContext());
  if (!context_) throw std::runtime_error("创建执行上下文失败");
}

void TrtDetector::setupBindings() {
  const int n = engine_->getNbBindings();
  binding_names_.resize(static_cast<size_t>(n));

  std::vector<int> outputs;
  for (int i = 0; i < n; ++i) {
    binding_names_[i] = engine_->getBindingName(i);
    if (engine_->bindingIsInput(i)) {
      if (in_idx_ < 0) in_idx_ = i;
    } else {
      outputs.push_back(i);
    }
  }
  if (in_idx_ < 0) throw std::runtime_error("引擎没有输入 binding");
  if (outputs.size() < 2) throw std::runtime_error("引擎输出 binding 少于 2 个（需要 score + map）");

  in_size_ = static_cast<int>(volume(engine_->getBindingDimensions(in_idx_)));

  // 用元素个数区分 score(1 个) 与 anomaly_map(1024 个)，不依赖输出顺序
  const int64_t v0 = volume(engine_->getBindingDimensions(outputs[0]));
  const int64_t v1 = volume(engine_->getBindingDimensions(outputs[1]));
  if (v0 == 1 && v1 > 1) {
    out_score_idx_ = outputs[0];
    out_map_idx_ = outputs[1];
  } else if (v1 == 1 && v0 > 1) {
    out_score_idx_ = outputs[1];
    out_map_idx_ = outputs[0];
  } else {
    out_score_idx_ = outputs[0];
    out_map_idx_ = outputs[1];
  }

  const nvinfer1::Dims md = engine_->getBindingDimensions(out_map_idx_);
  int64_t mw = size_, mh = size_;
  if (md.nbDims == 4) { mh = md.d[2]; mw = md.d[3]; }
  else if (md.nbDims == 3) { mh = md.d[1]; mw = md.d[2]; }
  else if (md.nbDims == 2) { mh = md.d[0]; mw = md.d[1]; }
  map_w_ = mw;
  map_h_ = mh;

  const size_t map_elems = static_cast<size_t>(mw * mh);
  host_in_.assign(static_cast<size_t>(in_size_), 0.f);
  host_score_.assign(1, 0.f);
  host_map_.assign(map_elems, 0.f);

  throwCuda(cudaMalloc(&dev_in_, host_in_.size() * sizeof(float)), "cudaMalloc(input)");
  throwCuda(cudaMalloc(&dev_score_, sizeof(float)), "cudaMalloc(score)");
  throwCuda(cudaMalloc(&dev_map_, map_elems * sizeof(float)), "cudaMalloc(map)");
  throwCuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
}

Prediction TrtDetector::predict(const cv::Mat& bgr, float thr_image, float thr_pixel) {
  const double t0 = nowMs();

  // ---------------- 预处理（与 Python to_tensor 一致） ----------------
  cv::Mat gray;
  if (bgr.channels() == 3) {
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = bgr;
  }
  cv::Mat resized;
  cv::resize(gray, resized, cv::Size(size_, size_), 0, 0, cv::INTER_AREA);
  const size_t plane = static_cast<size_t>(size_) * size_;
  for (int y = 0; y < size_; ++y) {
    const uint8_t* row = resized.ptr<uint8_t>(y);
    for (int x = 0; x < size_; ++x) {
      const float v = static_cast<float>(row[x]) / 255.0f;
      const size_t idx = static_cast<size_t>(y) * size_ + x;
      host_in_[idx] = v;
      host_in_[plane + idx] = v;
      host_in_[2 * plane + idx] = v;
    }
  }

  throwCuda(cudaMemcpyAsync(dev_in_, host_in_.data(), host_in_.size() * sizeof(float),
                            cudaMemcpyHostToDevice, stream_),
            "cudaMemcpyAsync(H2D)");

  const int max_idx = std::max(in_idx_, std::max(out_score_idx_, out_map_idx_));
  std::vector<void*> binding_ptrs(static_cast<size_t>(max_idx + 1), nullptr);
  binding_ptrs[in_idx_] = dev_in_;
  binding_ptrs[out_score_idx_] = dev_score_;
  binding_ptrs[out_map_idx_] = dev_map_;

  if (!context_->enqueueV2(binding_ptrs.data(), stream_, nullptr)) {
    throw std::runtime_error("TensorRT enqueueV2 失败");
  }

  throwCuda(cudaMemcpyAsync(host_score_.data(), dev_score_, sizeof(float),
                            cudaMemcpyDeviceToHost, stream_),
            "cudaMemcpyAsync(D2H score)");
  throwCuda(cudaMemcpyAsync(host_map_.data(), dev_map_, host_map_.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, stream_),
            "cudaMemcpyAsync(D2H map)");
  throwCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");

  // ---------------- 后处理（与 ORT 版逐位一致） ----------------
  Prediction pred;
  pred.score = host_score_[0];
  pred.is_defect = pred.score >= thr_image;
  pred.anomaly_map = cv::Mat(static_cast<int>(map_h_), static_cast<int>(map_w_), CV_32F,
                             host_map_.data())
                         .clone();

  double map_max = 0.0;
  cv::Point map_max_loc(0, 0);
  cv::minMaxLoc(pred.anomaly_map, nullptr, &map_max, nullptr, &map_max_loc);

  float effective = thr_pixel;
  if (pred.is_defect) {
    effective = std::min(thr_pixel, static_cast<float>(0.75 * map_max));
  }

  cv::Mat mask = cv::Mat::zeros(pred.anomaly_map.size(), CV_8U);
  int hot = 0;
  for (int y = 0; y < pred.anomaly_map.rows; ++y) {
    const float* src = pred.anomaly_map.ptr<float>(y);
    uint8_t* dst = mask.ptr<uint8_t>(y);
    for (int x = 0; x < pred.anomaly_map.cols; ++x) {
      if (src[x] >= effective) {
        dst[x] = 1;
        ++hot;
      }
    }
  }
  pred.hot_ratio = static_cast<float>(hot) / static_cast<float>(mask.total());

  cv::Mat labels, stats, centroids;
  const int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
  std::vector<Region> regions;
  for (int i = 1; i < n; ++i) {
    const int area = stats.at<int>(i, cv::CC_STAT_AREA);
    if (area < 12) continue;
    Region r;
    r.x = stats.at<int>(i, cv::CC_STAT_LEFT);
    r.y = stats.at<int>(i, cv::CC_STAT_TOP);
    r.w = stats.at<int>(i, cv::CC_STAT_WIDTH);
    r.h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
    r.area = area;
    regions.push_back(r);
  }
  std::sort(regions.begin(), regions.end(),
            [](const Region& a, const Region& b) { return a.area > b.area; });
  if (regions.size() > 8) regions.resize(8);

  if (pred.is_defect && regions.empty()) {
    const int mw_i = pred.anomaly_map.cols;
    const int mh_i = pred.anomaly_map.rows;
    const int half = std::max(6, static_cast<int>(0.08 * std::min(mw_i, mh_i)));
    Region r;
    r.x = std::max(0, std::min(mw_i - 1, map_max_loc.x - half));
    r.y = std::max(0, std::min(mh_i - 1, map_max_loc.y - half));
    r.w = std::min(2 * half, mw_i - r.x);
    r.h = std::min(2 * half, mh_i - r.y);
    r.area = r.w * r.h;
    regions.push_back(r);
  }
  pred.regions = std::move(regions);

  pred.elapsed_ms = nowMs() - t0;
  return pred;
}

}  // namespace nutsight
