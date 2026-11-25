#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NutSight 模型导出 / 量化 一条龙脚本。

    python export_quantize.py                       # 导出 release-512 → ONNX（opset 13）
    python export_quantize.py --check               # 导出后与 PyTorch 对拍（需要 onnxruntime）
    python export_quantize.py --quantize            # 额外产出 INT8 QDQ 模型（需要 onnxruntime）
    python export_quantize.py --model v20261103-004047-48280f-img640
    python export_quantize.py --model models/xxx/model.pt --out D:/tmp/x.onnx --opset 17

导出的 ONNX 可以直接喂给 trtexec 建 TensorRT 引擎（TX2 / Orin / x86 通用）。

说明
----
* 这个模型是 PatchCore：输入 1×3×H×W（RGB，已归一化到 [0,1]），
  输出 score(1×1) 与 anomaly_map(1×1×H×W)，TTA×4 已经固化在图里。
* ONNX 层的 INT8 量化和 TensorRT 的 INT8 是两回事：
  - 这里是 onnxruntime 的静态量化（QDQ），产物给 ORT 用；
  - 真正上板要的是 trtexec --int8 + 校准图，脚本末尾会打印现成命令。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nutsight.models.detector import PatchCoreDetector  # noqa: E402

DEFAULT_MODEL = "v20261031-035947-275274-release-512"
WEIGHTS_DIR = ROOT / "models" / "backbone" / "checkpoints"


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
class TupleWrapper(torch.nn.Module):
    """把 detector 的 dict 输出改成 (score, map) 元组，ONNX 导出更稳。"""

    def __init__(self, detector: PatchCoreDetector) -> None:
        super().__init__()
        self.detector = detector

    def forward(self, images: torch.Tensor):
        out = self.detector(images)
        return out["score"], out["map"]


def resolve_model(name_or_path: str) -> Path:
    """支持传版本名、相对路径、绝对路径。"""
    p = Path(name_or_path)
    if p.exists():
        return p.resolve()
    cand = ROOT / "models" / name_or_path / "model.pt"
    if cand.exists():
        return cand.resolve()
    raise SystemExit(f"[错误] 找不到模型：{name_or_path}")


def have(mod: str) -> bool:
    import importlib

    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


def load_detector(model_pt: Path) -> tuple[PatchCoreDetector, int]:
    if not WEIGHTS_DIR.exists():
        raise SystemExit(f"[错误] 找不到 backbone 权重目录：{WEIGHTS_DIR}")
    print(f"[1/4] 加载模型 {model_pt.parent.name}/model.pt")
    det = PatchCoreDetector.load(
        model_pt,
        pretrained=True,
        weights_dir=WEIGHTS_DIR,
        device="cpu",
    ).eval()
    size = int(det.image_size)
    print(f"      backbone={det.extractor.backbone}  输入={size}x{size}  TTA={det.tta_rotations}")
    return det, size


# --------------------------------------------------------------------------- #
# 导出
# --------------------------------------------------------------------------- #
def export_onnx(det: PatchCoreDetector, size: int, out: Path, opset: int) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.rand(1, 3, size, size)
    print(f"[2/4] 导出 ONNX → {out}")
    print(f"      opset={opset}  输入 1x3x{size}x{size}")

    kwargs = dict(
        input_names=["image"],
        output_names=["score", "anomaly_map"],
        opset_version=opset,
        do_constant_folding=True,
    )
    t0 = time.time()
    try:
        torch.onnx.export(TupleWrapper(det), dummy, str(out), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(TupleWrapper(det), dummy, str(out), **kwargs)
    dt = time.time() - t0

    mb = out.stat().st_size / 1024 / 1024
    print(f"      完成：{mb:.1f} MB，耗时 {dt:.1f}s")
    return out


# --------------------------------------------------------------------------- #
# 对拍
# --------------------------------------------------------------------------- #
def verify(onnx_path: Path, det: PatchCoreDetector, size: int, tol: float) -> bool:
    if not have("onnxruntime"):
        print("[跳过] 对拍需要 onnxruntime：pip install onnxruntime")
        return True

    import onnxruntime as ort

    print("[3/4] onnxruntime 对拍")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    gray = rng.normal(180, 30, (1, 1, size, size)).clip(0, 255) / 255.0
    x = np.concatenate([gray] * 3, axis=1).astype(np.float32)

    s_onnx, m_onnx = sess.run(None, {"image": x})
    with torch.no_grad():
        out = det(torch.from_numpy(x))
    s_pt = out["score"].numpy()
    m_pt = out["map"].numpy()

    ds = float(np.abs(s_onnx - s_pt).max())
    dm = float(np.abs(m_onnx - m_pt).max())
    ok = ds <= tol and dm <= tol
    flag = "OK" if ok else "不一致"
    print(f"      {flag}  score 误差={ds:.2e}  map 误差={dm:.2e}  容忍={tol:g}")
    print(f"      score: onnx={float(s_onnx.reshape(-1)[0]):.6f}  torch={float(s_pt.reshape(-1)[0]):.6f}")
    return ok


# --------------------------------------------------------------------------- #
# ONNX INT8 静态量化（QDQ）
# --------------------------------------------------------------------------- #
def quantize_int8(onnx_fp32: Path, onnx_out: Path, calib_dir: Path, size: int, n: int) -> Path:
    if not have("onnxruntime.quantization"):
        raise SystemExit(
            "[错误] 量化需要 onnxruntime：\n"
            "      pip install onnxruntime onnxruntime-tools"
        )

    import cv2
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    imgs = sorted(
        [p for p in calib_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
    )[:n]
    if not imgs:
        raise SystemExit(f"[错误] 校准目录里没有图片：{calib_dir}")
    print(f"[4/4] INT8 量化：校准集 {len(imgs)} 张 → {onnx_out.name}")

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self._it = iter(imgs)

        def get_next(self):
            try:
                p = next(self._it)
            except StopIteration:
                return None
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return self.get_next()
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            x = (img.astype(np.float32) / 255.0)[None, None]  # 1x1xHxW
            x = np.concatenate([x] * 3, axis=1).astype(np.float32)  # 1x3xHxW
            return {"image": x}

    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(onnx_fp32),
        model_output=str(onnx_out),
        calibration_data_reader=Reader(),
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    mb = onnx_out.stat().st_size / 1024 / 1024
    print(f"      完成：{mb:.1f} MB")
    return onnx_out


# --------------------------------------------------------------------------- #
# TensorRT 提示
# --------------------------------------------------------------------------- #
def print_trt_hint(onnx_path: Path, size: int) -> None:
    print("\n" + "=" * 72)
    print("下一步：在板子上建 TensorRT 引擎")
    print("=" * 72)
    print(f"""
# FP16（推荐先跑这个）
/usr/src/tensorrt/bin/trtexec \\
  --onnx={onnx_path.name} \\
  --saveEngine=model.fp16.engine \\
  --fp16 --workspace=1024

# INT8（把 calib/ 拷到板上，用 --int8-cache 目录）
/usr/src/tensorrt/bin/trtexec \\
  --onnx={onnx_path.name} \\
  --saveEngine=model.int8.engine \\
  --int8 --workspace=1024 \\
  --calib=...（trtexec 不支持自动校准，用项目里的 run.sh --int8）

提示：本项目的 run.sh --int8 已经内置 IInt8EntropyCalibrator2，
     直接跑 ./run.sh --int8 即可，不需要手工喂 calib。
""")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="NutSight pt → onnx（+ INT8 量化）")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="版本名或 model.pt 路径")
    ap.add_argument("--out", default=None, help="ONNX 输出路径（默认放在该版本目录下）")
    ap.add_argument("--opset", type=int, default=13, help="ONNX opset（TX2 的 TRT 8.2 用 13 最稳）")
    ap.add_argument("--check", action="store_true", help="导出后对拍（需要 onnxruntime）")
    ap.add_argument("--tol", type=float, default=1e-3, help="对拍容忍误差")
    ap.add_argument("--quantize", action="store_true", help="额外产出 INT8 QDQ 模型（需要 onnxruntime）")
    ap.add_argument("--calib-dir", default=None, help="校准图目录（默认 models/<版本>/../calib 或项目 calib/）")
    ap.add_argument("--calib-n", type=int, default=60, help="校准图张数")
    ap.add_argument("--quiet", action="store_true", help="不打印 TensorRT 提示")
    args = ap.parse_args()

    model_pt = resolve_model(args.model)
    det, size = load_detector(model_pt)

    out = Path(args.out) if args.out else model_pt.parent / "model.onnx"
    onnx_path = export_onnx(det, size, out, args.opset)

    ok = True
    if args.check:
        ok = verify(onnx_path, det, size, args.tol)

    if args.quantize:
        calib = Path(args.calib_dir) if args.calib_dir else (ROOT / "calib")
        q_out = onnx_path.with_name(onnx_path.stem + ".int8.onnx")
        quantize_int8(onnx_path, q_out, calib, size, args.calib_n)

    if not args.quiet:
        print_trt_hint(onnx_path, size)

    if not ok:
        print("\n[警告] ONNX 与 PyTorch 输出不一致，别拿去部署")
        return 2
    print("\n[完成]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
