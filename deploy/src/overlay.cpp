#include "overlay.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace nutsight {
namespace {

// 与 Python 端 utils/image.py:overlay_heatmap 保持一致：
//   alpha=0.85、threshold=0.4、vmax_quantile=0.9995
// 低分区几乎不着色，避免整图泛蓝。
constexpr double kAlpha = 0.85;
constexpr double kThreshold = 0.4;
constexpr double kVmaxQuantile = 0.9995;

double quantile(std::vector<float>& values, double q) {
  if (values.empty()) return 1.0;
  q = std::min(1.0, std::max(0.0, q));
  size_t idx = static_cast<size_t>(q * static_cast<double>(values.size() - 1));
  std::nth_element(values.begin(), values.begin() + idx, values.end());
  return static_cast<double>(values[idx]);
}

}  // namespace

cv::Mat overlayHeat(const cv::Mat& bgr, const cv::Mat& anomaly_map,
                    const std::vector<Region>& regions, int map_size, float pixel_thr) {
  cv::Mat vis = bgr.clone();
  const int w = vis.cols;
  const int h = vis.rows;

  // 1) 缩放
  cv::Mat heat;
  cv::resize(anomaly_map, heat, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
  if (heat.type() != CV_32F) heat.convertTo(heat, CV_32F);

  // 2) 决定归一化方式
  //    pixel_thr > 0：绝对刻度，vmax = pixel_thr * 2.5，正常图几乎不着色；
  //    pixel_thr <= 0：旧行为（每图 min-max + 0.9995 分位），会把弱响应也拉满。
  cv::Mat hm;
  if (pixel_thr > 0.0f) {
    const double vmax_abs = static_cast<double>(pixel_thr) * 2.5;
    heat.convertTo(hm, CV_32F, 1.0 / vmax_abs, 0.0);
  } else {
    double lo = 0.0, hi = 0.0;
    cv::minMaxLoc(heat, &lo, &hi);
    const double span = std::max(hi - lo, 1e-6);
    cv::Mat norm;
    heat.convertTo(norm, CV_32F, 1.0 / span, -lo / span);

    std::vector<float> samples;
    samples.reserve(static_cast<size_t>(norm.total()));
    for (int y = 0; y < norm.rows; ++y) {
      const float* row = norm.ptr<float>(y);
      samples.insert(samples.end(), row, row + norm.cols);
    }
    const double vmax = quantile(samples, kVmaxQuantile);
    const double scale = vmax > 1e-6 ? 1.0 / vmax : 0.0;
    norm.convertTo(hm, CV_32F, scale);
  }

  // 3) 背景不着色：低于 threshold 的置 0；叠加权重按 hm^1.5 * alpha
  cv::Mat u8(hm.size(), CV_8U);
  for (int y = 0; y < hm.rows; ++y) {
    float* src = hm.ptr<float>(y);
    uint8_t* dst = u8.ptr<uint8_t>(y);
    for (int x = 0; x < hm.cols; ++x) {
      float v = src[x];
      v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
      src[x] = v;  // 回写裁剪后的值，后面算权重用
      dst[x] = static_cast<uint8_t>(v * 255.0f);
    }
  }

  cv::Mat color;
  cv::applyColorMap(u8, color, cv::COLORMAP_JET);  // BGR，与 base 同色序

  cv::Mat out(vis.size(), vis.type());
  for (int y = 0; y < h; ++y) {
    const cv::Vec3b* base = vis.ptr<cv::Vec3b>(y);
    const cv::Vec3b* col = color.ptr<cv::Vec3b>(y);
    const float* hrow = hm.ptr<float>(y);
    cv::Vec3b* dst = out.ptr<cv::Vec3b>(y);
    for (int x = 0; x < w; ++x) {
      const float hmv = hrow[x] >= static_cast<float>(kThreshold) ? hrow[x] : 0.f;
      const float weight = static_cast<float>(std::pow(hmv, 1.5) * kAlpha);
      for (int c = 0; c < 3; ++c) {
        const float v = base[x][c] * (1.0f - weight) + col[x][c] * weight;
        dst[x][c] = static_cast<uint8_t>(v < 0.f ? 0.f : (v > 255.f ? 255.f : v));
      }
    }
  }

  // 4) 缺陷框（模型输出坐标 → 原图坐标）
  const double sx = static_cast<double>(w) / std::max(1, map_size);
  const double sy = static_cast<double>(h) / std::max(1, map_size);
  const int thickness = std::max(2, w / 400);
  for (size_t i = 0; i < regions.size(); ++i) {
    const auto& r = regions[i];
    const int x0 = static_cast<int>(r.x * sx);
    const int y0 = static_cast<int>(r.y * sy);
    const int x1 = static_cast<int>((r.x + r.w) * sx);
    const int y1 = static_cast<int>((r.y + r.h) * sy);
    cv::rectangle(out, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 0, 255), thickness);
    // 只给最大的那个框打序号，避免多个小框的标签叠在一起
    if (i == 0) {
      char tag[16];
      std::snprintf(tag, sizeof(tag), "NG#%d", static_cast<int>(i) + 1);
      const double fs = std::max(0.5, w / 900.0);
      int bl = 0;
      const cv::Size ts = cv::getTextSize(tag, cv::FONT_HERSHEY_SIMPLEX, fs, 1, &bl);
      const int ty = std::max(ts.height + 4, y0 - 4);
      cv::rectangle(out, cv::Point(x0, ty - ts.height - 4),
                    cv::Point(x0 + ts.width + 8, ty + 4), cv::Scalar(0, 0, 255), cv::FILLED);
      cv::putText(out, tag, cv::Point(x0 + 4, ty), cv::FONT_HERSHEY_SIMPLEX, fs,
                  cv::Scalar(255, 255, 255), 1);
    }
  }
  return out;
}

void drawStatus(cv::Mat& img, bool is_defect, float score, double ms) {
  char text[96];
  std::snprintf(text, sizeof(text), "%s %.4f  %.0fms", is_defect ? "NG" : "OK", score, ms);
  const cv::Scalar color = is_defect ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 200, 0);

  int baseline = 0;
  const double scale = 0.9;
  const cv::Size sz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, 2, &baseline);
  const int tw = sz.width;
  const int th = sz.height;

  cv::rectangle(img, cv::Point(8, 8), cv::Point(8 + tw + 16, 8 + th + 16),
                cv::Scalar(20, 22, 28), cv::FILLED);
  cv::putText(img, text, cv::Point(16, 8 + th + 8), cv::FONT_HERSHEY_SIMPLEX, scale, color, 2);
}

}  // namespace nutsight
