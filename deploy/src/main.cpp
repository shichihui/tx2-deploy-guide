#include "trt_detector.h"
#include "imageio.h"
#include "overlay.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

struct Args {
  std::string model;
  std::string engine;
  std::string precision{"fp16"};
  std::string calib_dir;
  std::string image;
  std::string dir;
  std::string video;
  std::string save_dir;
  std::string json;
  std::string csv;
  float thr_image = -1.f;
  float thr_pixel = -1.f;
  int threads = 0;
  int stride = 1;
  int max_seconds = 0;
  int device = 0;
  int workspace_mb = 1024;
  bool rebuild = false;
  bool quiet = false;
};

/// 从 model.json 里抠 thresholds.image / thresholds.pixel（不引入 JSON 库）。
bool readThresholds(const std::string& path, float& image, float& pixel) {
  std::ifstream fin(path);
  if (!fin) return false;
  std::stringstream ss;
  ss << fin.rdbuf();
  const std::string text = ss.str();

  auto extract = [&text](const char* key, float& out) -> bool {
    const std::string k = std::string("\"") + key + "\"";
    size_t pos = text.find(k);
    if (pos == std::string::npos) return false;
    pos = text.find(':', pos + k.size());
    if (pos == std::string::npos) return false;
    ++pos;
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t')) ++pos;
    char* end = nullptr;
    const double v = std::strtod(text.c_str() + pos, &end);
    if (end == text.c_str() + pos) return false;
    out = static_cast<float>(v);
    return true;
  };

  const size_t thr_pos = text.find("\"thresholds\"");
  bool ok = false;
  if (thr_pos != std::string::npos) {
    // 只在 thresholds 之后的片段里找，避免命中别处的 image_auroc
    const std::string sub = text.substr(thr_pos);
    auto extract_in = [&sub](const char* key, float& out) -> bool {
      const std::string k = std::string("\"") + key + "\"";
      size_t pos = sub.find(k);
      if (pos == std::string::npos) return false;
      pos = sub.find(':', pos + k.size());
      if (pos == std::string::npos) return false;
      ++pos;
      while (pos < sub.size() && (sub[pos] == ' ' || sub[pos] == '\t')) ++pos;
      char* end = nullptr;
      const double v = std::strtod(sub.c_str() + pos, &end);
      if (end == sub.c_str() + pos) return false;
      out = static_cast<float>(v);
      return true;
    };
    bool got_i = extract_in("image", image);
    bool got_p = extract_in("pixel", pixel);
    ok = got_i || got_p;
  }
  if (!ok) {
    bool got_i = extract("image", image);
    bool got_p = extract("pixel", pixel);
    ok = got_i || got_p;
  }
  return ok;
}

std::string jsonEscape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += c;
    }
  }
  return out;
}

std::string regionsJson(const std::vector<nutsight::Region>& regions) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < regions.size(); ++i) {
    if (i) os << ",";
    const auto& r = regions[i];
    os << "{\"x\":" << r.x << ",\"y\":" << r.y << ",\"w\":" << r.w << ",\"h\":" << r.h
       << ",\"area\":" << r.area << "}";
  }
  os << "]";
  return os.str();
}

std::string regionsCsv(const std::vector<nutsight::Region>& regions) {
  std::ostringstream os;
  for (size_t i = 0; i < regions.size(); ++i) {
    if (i) os << ";";
    const auto& r = regions[i];
    os << r.x << ":" << r.y << ":" << r.w << ":" << r.h;
  }
  return os.str();
}

void printUsage() {
  std::printf(
      "NutSight 板端推理（C++ / TensorRT GPU）\n\n"
      "用法:\n"
      "  nutsight-cli --image a.jpg\n"
      "  nutsight-cli --dir imgs/ --save-dir out/\n"
      "  nutsight-cli --video line.mp4 --save-dir out/\n"
      "  nutsight-cli --precision int8 --calib-dir calib/ --image a.jpg   # 首次会构建 INT8 引擎\n\n"
      "模型与精度:\n"
      "  --model <path>      model.onnx（默认：可执行文件同目录）\n"
      "  --engine <path>     .engine 文件（默认 model.<precision>.engine，不存在则用 ONNX 构建）\n"
      "  --precision <p>     fp32 | fp16 | int8（默认 fp16）\n"
      "  --calib-dir <path>  INT8 校准图目录（正常样本，默认 calib/）\n"
      "  --rebuild           即使引擎已存在也强制重建\n"
      "  --device <n>        CUDA 设备号（默认 0）\n"
      "  --workspace <mb>    构建期的 workspace 上限（默认 1024）\n\n"
      "输入:\n"
      "  --image <path>      单张图片\n"
      "  --dir <path>        批量目录（递归）\n"
      "  --video <path>      视频文件\n"
      "  --save-dir <path>   输出目录（标注图/标注视频/CSV）\n"
      "  --json <path>       批量图片结果写入 JSON\n"
      "  --csv <path>        视频逐帧结果写入 CSV（默认 save-dir/report.csv）\n"
      "  --thr-image <f>     覆盖图像阈值\n"
      "  --thr-pixel <f>     覆盖像素阈值\n"
      "  --stride <n>        视频每 n 帧检测 1 帧（默认 1）\n"
      "  --max-seconds <n>   只处理前 n 秒（0=全片）\n"
      "  --quiet             只打印汇总\n");
}

bool parseArgs(int argc, char** argv, Args& args) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "[error] %s 缺少参数值\n", name);
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--model") args.model = next("--model");
    else if (a == "--engine") args.engine = next("--engine");
    else if (a == "--precision") args.precision = next("--precision");
    else if (a == "--calib-dir") args.calib_dir = next("--calib-dir");
    else if (a == "--rebuild") args.rebuild = true;
    else if (a == "--device") args.device = std::atoi(next("--device").c_str());
    else if (a == "--workspace") args.workspace_mb = std::atoi(next("--workspace").c_str());
    else if (a == "--image") args.image = next("--image");
    else if (a == "--dir") args.dir = next("--dir");
    else if (a == "--video") args.video = next("--video");
    else if (a == "--save-dir") args.save_dir = next("--save-dir");
    else if (a == "--json") args.json = next("--json");
    else if (a == "--csv") args.csv = next("--csv");
    else if (a == "--thr-image") args.thr_image = std::strtof(next("--thr-image").c_str(), nullptr);
    else if (a == "--thr-pixel") args.thr_pixel = std::strtof(next("--thr-pixel").c_str(), nullptr);
    else if (a == "--threads") args.threads = std::atoi(next("--threads").c_str());
    else if (a == "--stride") args.stride = std::max(1, std::atoi(next("--stride").c_str()));
    else if (a == "--max-seconds") args.max_seconds = std::atoi(next("--max-seconds").c_str());
    else if (a == "--quiet" || a == "-q") args.quiet = true;
    else if (a == "--help" || a == "-h") { printUsage(); std::exit(0); }
    else {
      std::fprintf(stderr, "[error] 未知参数: %s\n", a.c_str());
      printUsage();
      return false;
    }
  }
  return true;
}

bool fileExists(const std::string& p) {
  std::error_code ec;
  return fs::exists(p, ec) && !fs::is_directory(p, ec);
}

std::string defaultModelPath() {
  std::string dir = ".";
#if defined(__linux__)
  char exe[4096];
  const ssize_t n = ::readlink("/proc/self/exe", exe, sizeof(exe) - 1);
  if (n > 0) {
    exe[n] = '\0';
    const std::string p(exe);
    const size_t slash = p.find_last_of('/');
    if (slash != std::string::npos) dir = p.substr(0, slash);
  }
#elif defined(_WIN32)
  char exe[4096];
  const DWORD n = ::GetModuleFileNameA(nullptr, exe, sizeof(exe));
  if (n > 0) {
    const std::string p(exe, n);
    const size_t slash = p.find_last_of("\\/");
    if (slash != std::string::npos) dir = p.substr(0, slash);
  }
#endif
  // 依次尝试：可执行文件同目录 → ../model → 当前目录
  const std::string candidates[] = {
      dir + "/model.onnx", dir + "/../model/model.onnx", "model.onnx",
  };
  for (const auto& c : candidates) {
    if (fileExists(c)) return c;
  }
  return dir + "/model.onnx";
}

void ensureDir(const std::string& dir) {
  if (dir.empty()) return;
  std::error_code ec;
  fs::create_directories(dir, ec);
}

int runImage(nutsight::TrtDetector& det, const Args& args, float thr_image, float thr_pixel) {
  const cv::Mat bgr = nutsight::loadBgr(args.image);
  const auto pred = det.predict(bgr, thr_image, thr_pixel);
  const char* verdict = pred.is_defect ? "NG 缺陷" : "OK 合格";
  std::printf("%-28s %s  score=%.4f (阈值 %.4f)  热区 %.2f%%  %zu 个区域  %.0f ms\n",
              nutsight::baseName(args.image).c_str(), verdict, pred.score, thr_image,
              pred.hot_ratio * 100.0f, pred.regions.size(), pred.elapsed_ms);
  if (!args.save_dir.empty()) {
    ensureDir(args.save_dir);
    const cv::Mat vis = nutsight::overlayHeat(bgr, pred.anomaly_map, pred.regions, pred.anomaly_map.cols, thr_pixel);
    const std::string out = args.save_dir + "/" + nutsight::stemOf(args.image) + "_result.jpg";
    if (nutsight::saveImage(out, vis)) {
      std::printf("%-28s 结果图 -> %s\n", "", out.c_str());
    } else {
      std::fprintf(stderr, "[warn] 结果图写出失败: %s\n", out.c_str());
    }
  }
  return pred.is_defect ? 1 : 0;
}

int runDir(nutsight::TrtDetector& det, const Args& args, float thr_image, float thr_pixel) {
  const auto files = nutsight::listImages(args.dir);
  if (files.empty()) {
    std::fprintf(stderr, "[error] 目录里没有图片: %s\n", args.dir.c_str());
    return 2;
  }
  std::printf("共 %zu 张，模型加载 %.0f ms\n\n", files.size(), det.loadMs());
  ensureDir(args.save_dir);

  std::ostringstream json;
  json << "[\n";
  size_t ng = 0;
  double total_ms = 0.0;
  size_t idx = 0;
  for (const auto& path : files) {
    try {
      const cv::Mat bgr = nutsight::loadBgr(path);
      const auto pred = det.predict(bgr, thr_image, thr_pixel);
      total_ms += pred.elapsed_ms;
      if (pred.is_defect) ++ng;
      if (!args.quiet) {
        std::printf("%-28s %s  score=%.4f  热区 %.2f%%  %.0f ms\n",
                    nutsight::baseName(path).c_str(), pred.is_defect ? "NG" : "OK",
                    pred.score, pred.hot_ratio * 100.0f, pred.elapsed_ms);
      }
      if (!args.save_dir.empty()) {
        const cv::Mat vis = nutsight::overlayHeat(bgr, pred.anomaly_map, pred.regions, pred.anomaly_map.cols, thr_pixel);
        nutsight::saveImage(args.save_dir + "/" + nutsight::stemOf(path) + "_result.jpg", vis);
      }
      if (!args.json.empty()) {
        if (idx) json << ",\n";
        json << "  {\"file\":\"" << jsonEscape(path) << "\",\"score\":" << std::fixed
             << std::setprecision(6) << pred.score << ",\"is_defect\":"
             << (pred.is_defect ? "true" : "false") << ",\"threshold\":" << thr_image
             << ",\"hot_ratio\":" << pred.hot_ratio << ",\"regions\":" << regionsJson(pred.regions)
             << ",\"elapsed_ms\":" << pred.elapsed_ms << "}";
      }
      ++idx;
    } catch (const std::exception& exc) {
      std::fprintf(stderr, "[warn] %s 失败: %s\n", path.c_str(), exc.what());
    }
  }
  json << "\n]\n";

  const size_t n = idx ? idx : 1;
  std::printf("\n合计 %zu 张：NG %zu / OK %zu，平均 %.0f ms/张\n", idx, ng, idx - ng,
              total_ms / n);
  if (!args.json.empty()) {
    std::ofstream fout(args.json);
    if (fout) {
      fout << json.str();
      std::printf("明细已写入 %s\n", args.json.c_str());
    } else {
      std::fprintf(stderr, "[warn] 写 JSON 失败: %s\n", args.json.c_str());
    }
  }
  return 0;
}

int runVideo(nutsight::TrtDetector& det, const Args& args, float thr_image, float thr_pixel) {
  cv::VideoCapture cap(args.video);
  if (!cap.isOpened()) {
    std::fprintf(stderr, "[error] 打不开视频: %s\n", args.video.c_str());
    return 2;
  }
  const double fps = cap.get(cv::CAP_PROP_FPS) > 1.0 ? cap.get(cv::CAP_PROP_FPS) : 25.0;
  const int total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
  const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

  ensureDir(args.save_dir);
  const std::string out_dir = args.save_dir.empty() ? "." : args.save_dir;
  const std::string out_video = out_dir + "/" + nutsight::stemOf(args.video) + "_annotated.mp4";
  const std::string csv_path =
      args.csv.empty() ? out_dir + "/report.csv" : args.csv;

  cv::VideoWriter writer(out_video, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                         cv::Size(width, height));
  if (!writer.isOpened()) {
    std::fprintf(stderr, "[error] 无法创建输出视频: %s\n", out_video.c_str());
    return 2;
  }

  std::ofstream csv(csv_path, std::ios::binary);
  if (csv) {
    csv << "\xEF\xBB\xBF" << "frame,time_s,score,is_defect,hot_ratio,regions\n";
  }

  std::printf("视频 %s  %dx%d  %.1f fps  %d 帧  stride=%d\n", nutsight::baseName(args.video).c_str(),
              width, height, fps, total, args.stride);

  const int max_frames = args.max_seconds > 0 ? static_cast<int>(args.max_seconds * fps) : 0;
  int frame_no = 0;
  int analyzed = 0;
  int ng = 0;
  float max_score = 0.f;
  double sum_score = 0.0;
  const double t_start = cv::getTickCount();

  cv::Mat frame;
  cv::Mat last_vis;  // stride>1 时重复上一张标注帧，避免输出视频一闪一闪
  while (cap.read(frame)) {
    if (!frame.empty() && (frame_no % args.stride) == 0) {
      const auto pred = det.predict(frame, thr_image, thr_pixel);
      cv::Mat vis = nutsight::overlayHeat(frame, pred.anomaly_map, pred.regions, pred.anomaly_map.cols, thr_pixel);
      nutsight::drawStatus(vis, pred.is_defect, pred.score, pred.elapsed_ms);
      last_vis = vis;
      writer.write(vis);

      ++analyzed;
      if (pred.is_defect) ++ng;
      max_score = std::max(max_score, pred.score);
      sum_score += pred.score;
      if (csv) {
        csv << frame_no << "," << std::fixed << std::setprecision(3) << (frame_no / fps) << ","
            << std::setprecision(6) << pred.score << "," << (pred.is_defect ? 1 : 0) << ","
            << pred.hot_ratio << ",\"" << regionsCsv(pred.regions) << "\"\n";
        csv.flush();
      }
      if (!args.quiet && analyzed % 30 == 0) {
        std::printf("  已处理 %d 帧（NG %d，最高分 %.4f）\n", analyzed, ng, max_score);
        std::fflush(stdout);
      }
    } else {
      writer.write(last_vis.empty() ? frame : last_vis);
    }
    ++frame_no;
    if (max_frames > 0 && frame_no >= max_frames) break;
  }

  cap.release();
  writer.release();
  if (csv) csv.close();

  const double elapsed_s = (cv::getTickCount() - t_start) / cv::getTickFrequency();
  std::printf("\n完成：%d/%d 帧已检测，NG %d 帧（%.1f%%），最高分 %.4f，平均 %.4f，用时 %.1f s（%.2f s/帧）\n",
              analyzed, frame_no, ng, analyzed ? 100.0 * ng / analyzed : 0.0, max_score,
              analyzed ? sum_score / analyzed : 0.0, elapsed_s,
              analyzed ? elapsed_s / analyzed : 0.0);
  std::printf("标注视频 -> %s\n逐帧结果 -> %s\n", out_video.c_str(), csv_path.c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!parseArgs(argc, argv, args)) return 2;

  if (args.image.empty() && args.dir.empty() && args.video.empty()) {
    printUsage();
    return 2;
  }
  if (args.model.empty()) args.model = defaultModelPath();

  float thr_image = args.thr_image;
  float thr_pixel = args.thr_pixel;
  std::string meta_path = args.model;
  const size_t dot = meta_path.rfind('.');
  if (dot != std::string::npos) meta_path = meta_path.substr(0, dot);
  meta_path += ".json";

  bool from_meta = false;
  if (thr_image < 0.f || thr_pixel < 0.f) {
    float img_v = 0.0765f;
    float pix_v = 0.0902f;
    if (readThresholds(meta_path, img_v, pix_v)) {
      from_meta = true;
      if (thr_image < 0.f) thr_image = img_v;
      if (thr_pixel < 0.f) thr_pixel = pix_v;
    } else {
      if (thr_image < 0.f) thr_image = 0.0765393f;
      if (thr_pixel < 0.f) thr_pixel = 0.0902226f;
    }
  }

  try {
    nutsight::TrtOptions opt;
    opt.onnx_path = args.model;
    opt.engine_path = args.engine;
    opt.calib_dir = args.calib_dir;
    opt.precision = nutsight::parsePrecision(args.precision);
    opt.image_size = 512;
    opt.workspace_mb = args.workspace_mb;
    opt.device = args.device;
    opt.rebuild = args.rebuild;

    if (opt.precision == nutsight::Precision::kINT8 && opt.calib_dir.empty()) {
      std::error_code ec;
      if (fs::is_directory("calib", ec)) opt.calib_dir = "calib";
    }

    nutsight::TrtDetector det(opt);
    std::printf("模型 %s 精度 %s 引擎 %s，加载 %.0f ms，阈值 image=%.4f pixel=%.4f%s\n",
                nutsight::baseName(det.enginePath()).c_str(), det.precision(),
                det.builtNow() ? "（本次新建）" : "（复用）", det.loadMs(), thr_image, thr_pixel,
                from_meta ? "（来自 model.json）" : "（内置默认）");

    if (!args.image.empty()) return runImage(det, args, thr_image, thr_pixel);
    if (!args.dir.empty()) return runDir(det, args, thr_image, thr_pixel);
    return runVideo(det, args, thr_image, thr_pixel);
  } catch (const std::exception& exc) {
    std::fprintf(stderr, "[error] %s\n", exc.what());
    return 1;
  }
}
