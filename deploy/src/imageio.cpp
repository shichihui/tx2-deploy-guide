#include "imageio.h"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace nutsight {
namespace {

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

}  // namespace

cv::Mat loadBgr(const std::string& path) {
  std::ifstream fin(path, std::ios::binary);
  if (!fin) {
    throw std::runtime_error("打不开文件: " + path);
  }
  std::vector<unsigned char> buf((std::istreambuf_iterator<char>(fin)),
                                 std::istreambuf_iterator<char>());
  if (buf.empty()) {
    throw std::runtime_error("文件是空的: " + path);
  }
  cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
  if (img.empty()) {
    throw std::runtime_error("无法解码图片: " + path);
  }
  return img;
}

bool saveImage(const std::string& path, const cv::Mat& img) {
  const std::string ext = lower(fs::path(path).extension().string());
  std::vector<unsigned char> buf;
  if (ext == ".png") {
    if (!cv::imencode(".png", img, buf)) return false;
  } else if (ext == ".bmp") {
    if (!cv::imencode(".bmp", img, buf)) return false;
  } else {
    std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, 92};
    if (!cv::imencode(".jpg", img, buf, params)) return false;
  }
  std::ofstream fout(path, std::ios::binary);
  if (!fout) return false;
  fout.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
  return fout.good();
}

bool looksLikeImage(const std::string& path) {
  static const std::vector<std::string> exts{".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"};
  const std::string ext = lower(fs::path(path).extension().string());
  return std::find(exts.begin(), exts.end(), ext) != exts.end();
}

bool looksLikeVideo(const std::string& path) {
  static const std::vector<std::string> exts{".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"};
  const std::string ext = lower(fs::path(path).extension().string());
  return std::find(exts.begin(), exts.end(), ext) != exts.end();
}

std::vector<std::string> listImages(const std::string& dir) {
  std::vector<std::string> files;
  std::error_code ec;
  for (auto it = fs::recursive_directory_iterator(dir, fs::directory_options::skip_permission_denied, ec);
       it != fs::recursive_directory_iterator(); it.increment(ec)) {
    if (ec) break;
    if (!it->is_regular_file(ec)) continue;
    const std::string p = it->path().string();
    if (looksLikeImage(p)) files.push_back(p);
  }
  std::sort(files.begin(), files.end());
  return files;
}

std::string stemOf(const std::string& path) { return fs::path(path).stem().string(); }

std::string baseName(const std::string& path) { return fs::path(path).filename().string(); }

}  // namespace nutsight
