#pragma once

#include "trt_detector.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace nutsight {

/// 把异常热图叠到原图（BGR）上，并画缺陷框。
/// pixel_thr > 0 时按"绝对刻度"着色：heat/vmax 中 vmax = pixel_thr * 2.5，
///   正常图（峰值远低于 pixel_thr）几乎不着色，缺陷图只在高响应处泛红；
/// pixel_thr <= 0 时退回旧行为（每图 min-max 归一化 + 0.9995 分位上限），
///   这种归一化会把弱响应也拉成满屏红色，容易让正常图看起来"全是缺陷"。
/// map_size 传模型输出热图的宽度（异常图坐标 → 原图坐标的换算基准）。
cv::Mat overlayHeat(const cv::Mat& bgr, const cv::Mat& anomaly_map,
                    const std::vector<Region>& regions, int map_size,
                    float pixel_thr = 0.0f);

/// 左上角状态条："NG 0.0852" / "OK 0.0455"。
void drawStatus(cv::Mat& img, bool is_defect, float score, double ms);

}  // namespace nutsight
