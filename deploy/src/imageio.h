#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace nutsight {

/// 读图（BGR）。用字节流解码以兼容 UTF-8 / 中文路径，与 Python 端 np.fromfile+imdecode 等价。
cv::Mat loadBgr(const std::string& path);

/// 写图（按扩展名编码）。同样走字节流，兼容中文路径。
bool saveImage(const std::string& path, const cv::Mat& img);

/// 递归列出目录下的图片文件（按路径排序）。
std::vector<std::string> listImages(const std::string& dir);

/// 判断扩展名是否像图片。
bool looksLikeImage(const std::string& path);

/// 判断扩展名是否像视频。
bool looksLikeVideo(const std::string& path);

std::string stemOf(const std::string& path);
std::string baseName(const std::string& path);

}  // namespace nutsight
