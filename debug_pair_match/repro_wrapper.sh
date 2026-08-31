#!/usr/bin/env bash
# ============================================================================
# Path A — "wrapper" (the reference): HF computes vision in the MAIN process and
# splices the features into inputs_embeds via the checkpoint's own model.chat();
# only the decode goes to a stock vLLM 0.21.0 engine loaded with the TEXT-ONLY
# extracted Qwen3, fed prompt_embeds.
#
# Expected: prismm_bench_pair_match = 78.65, bit-identical across reruns.
#
# Placeholders:
#   <WRAPPER>  lmms-eval checkout with lmms_eval/models/simple/internvl3_5_KTH.py
#   <CKPT>     same checkpoint as repro_fork.sh
#   <PY>       python from the wrapper env (stock vLLM 0.21.0 -- NOT the fork env)
#   <GPU>      CUDA device index
# ============================================================================
set -uo pipefail

WRAPPER="${WRAPPER:-<WRAPPER>}"
CKPT="${CKPT:-<CKPT>}"
PY="${PY:-<PY>}"
GPU="${GPU:-<GPU>}"
TAG="${TAG:-wrapper_pairmatch}"

[ -f "$CKPT/config.json" ] || { echo "[err] no config.json in $CKPT"; exit 1; }

export LOCAL_BRANCH="${LOCAL_BRANCH:-global_ca}"

# Decode path selector. These are ENV ONLY -- they are recorded nowhere in results.json,
# so the output folder name is the only record of which path produced a number.
export KTH_USE_HF="${KTH_USE_HF:-0}"        # 1 = no vLLM at all (scores 77.60)
export KTH_VLLM_ASYNC="${KTH_VLLM_ASYNC:-0}" # 0 = the reference config

# Batch-invariant deterministic decode kernels. Worth +4.17 points here:
#   78.65 with, 74.48 without. run_eval.sh turns this on whenever GEN_BATCH > 1.
export GEN_BATCH="${GEN_BATCH:-8}"
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"

# vision runs in HF in this process, so vLLM gets a much smaller share than in Path B.
LLM_GPU_MEM="${LLM_GPU_MEM:-0.48}"
LLM_MAXLEN="${LLM_MAXLEN:-40960}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

env \
  TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING \
  LMMS_LOG_FULL=1 \
  PYTHONPATH="$WRAPPER/src/lmms-eval" \
  CUDA_VISIBLE_DEVICES="$GPU" \
  "$PY" -m lmms_eval \
    --model internvl3_5_KTH \
    --model_args "pretrained=$CKPT,llm_gpu_memory_utilization=$LLM_GPU_MEM,llm_max_model_len=$LLM_MAXLEN" \
    --tasks prismm_bench_pair_match \
    --batch_size 1 --log_samples \
    --output_path "$WRAPPER/results/$TAG"

echo "[done] samples: $WRAPPER/results/$TAG"
echo "[note] rescore from raw samples before comparing (see repro_fork.sh)."
