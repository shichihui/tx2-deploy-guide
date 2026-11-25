#!/usr/bin/env bash
# NutSight GPU 版（TensorRT）—— 统一入口
#   ./run.sh                        # 跑 samples（FP16，首次自动建引擎）
#   ./run.sh --int8                 # INT8（用 calib/ 校准，首次会量化）
#   ./run.sh --fp32                 # 全精度
#   ./run.sh --image x.png          # 单张
#   ./run.sh --dir 目录 --save-dir out
#   ./run.sh --video x.mp4 --stride 2
#   ./run.sh --rebuild --int8       # 强制重建引擎
set -u

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here" || { echo "[错误] 进不去 $here"; exit 1; }

PREC="fp16"
CALIB="$here/calib"
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --int8)   PREC="int8"; shift ;;
    --fp16)   PREC="fp16"; shift ;;
    --fp32)   PREC="fp32"; shift ;;
    --calib-dir) CALIB="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

EXE=""
for cand in "$here/build/nutsight-cli" "$here/bin/nutsight-cli"; do
  [ -x "$cand" ] && EXE="$cand" && break
done

# 源代码比可执行文件新 → 自动重编
if [ -n "$EXE" ]; then
  while IFS= read -r f; do
    if [ "$f" -nt "$EXE" ]; then EXE=""; break; fi
  done < <(find "$here/src" -maxdepth 1 -type f \( -name '*.cpp' -o -name '*.h' \) 2>/dev/null)
fi

if [ -z "$EXE" ]; then
  echo "== 没有找到已编译的程序，先编译一次（约 1~3 分钟，只需一次） =="
  bash "$here/build.sh" || { echo "[错误] 编译失败，把上面最后几行报错发我"; exit 1; }
  EXE="$here/build/nutsight-cli"
fi

# 运行库路径（TRT / cudart 在系统目录，lib-deps 里是软链备份）
export LD_LIBRARY_PATH="$here/build/lib-deps:$here/lib:${LD_LIBRARY_PATH:-}"

MODEL="$here/model.onnx"
[ -f "$MODEL" ] || MODEL="$here/model/model.onnx"
ENGINE="$here/model.$PREC.engine"

case "$PREC" in
  int8) echo "== 精度：INT8（校准集 $CALIB） ==" ;;
  fp32) echo "== 精度：FP32 ==" ;;
  *)    echo "== 精度：FP16（推荐，速度约 FP32 的 2 倍，分数几乎不变） ==" ;;
esac

exec "$EXE" --model "$MODEL" --engine "$ENGINE" --precision "$PREC" \
     --calib-dir "$CALIB" \
     "${ARGS[@]:---dir $here/samples --save-dir $here/out}"
