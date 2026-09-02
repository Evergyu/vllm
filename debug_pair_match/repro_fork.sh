#!/usr/bin/env bash
# ============================================================================
# Path B — "fork": vision runs INSIDE vLLM EngineCore.
#
# Expected: prismm_bench_pair_match ~= 63-64  (the number under investigation)
# Reference (Path A, repro_wrapper.sh):        78.65
#
# Requires a GPU and the checkpoint. 192 questions, a few minutes on one 80GB card.
#
# Placeholders to fill in:
#   <FORK>  lmms-eval checkout that has lmms_eval/models/simple/vllm.py with the
#           <image>-placeholder splicing (see DEBUG_PAIR_MATCH.md 4.7)
#   <CKPT>  InternVL3.5-8B + KTH checkpoint directory (must contain config.json)
#   <PY>    python from the env where THIS vLLM build is installed
#   <GPU>   CUDA device index
# ============================================================================
set -uo pipefail

FORK="${FORK:-<FORK>}"
CKPT="${CKPT:-<CKPT>}"
PY="${PY:-<PY>}"
GPU="${GPU:-<GPU>}"
TAG="${TAG:-fork_pairmatch}"
H="$(hostname)"

[ -f "$CKPT/config.json" ] || { echo "[err] no config.json in $CKPT"; exit 1; }

# LOCAL_BRANCH selects the KTH spatial branch and therefore the extra tokens per tile.
# global_ca -> 12 extra -> 268 image tokens per tile. A wrong value silently changes
# the placeholder count; the shim raises rather than guessing on an unknown value.
export LOCAL_BRANCH="${LOCAL_BRANCH:-global_ca}"

# The checkpoint's chat_template.jinja has NO system message, but the model was SFT'd
# with the system_message baked into config.template (internvl2_5). The HF/wrapper path
# gets it via get_conv_template(); the fork only sees jinja. Generate a template with
# the system message injected -- per checkpoint, never hardcoded.
#   python eval/_src/make_fork_template.py <CKPT> --verify
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$FORK/templates/system.jinja}"  # override per checkpoint

# Toggle these to reproduce the knob sweep in DEBUG_PAIR_MATCH.md 2 (all within noise):
#   export VLLM_BATCH_INVARIANT=1     # -> 64.06
#   ...,enforce_eager=True            # -> 66.15
#   ...,enable_prefix_caching=False   # -> 64.06
#   ...,max_num_seqs=1                # -> 64.06

env \
  HF_HOME="${HF_HOME:-$HOME/hf_cache}" \
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  LMMS_LOG_FULL=1 \
  PYTHONPATH="$FORK/src/lmms-eval" \
  TMPDIR="$FORK/.tmpcache.$H/tmp" \
  TRITON_CACHE_DIR="$FORK/.tmpcache.$H/triton" \
  TORCHINDUCTOR_CACHE_DIR="$FORK/.tmpcache.$H/inductor" \
  VLLM_CACHE_ROOT="$FORK/.tmpcache.$H/vllm" \
  CUDA_VISIBLE_DEVICES="$GPU" \
  "$PY" -m lmms_eval \
    --model vllm \
    --model_args "model=$CKPT,gpu_memory_utilization=0.85,max_model_len=40960,trust_remote_code=True,chat_template=$CHAT_TEMPLATE" \
    --tasks prismm_bench_pair_match \
    --batch_size 8 --log_samples \
    --output_path "$FORK/results/$TAG"

echo "[done] samples: $FORK/results/$TAG"
echo "[note] the raw metric in results.json is NOT the published number -- responses"
echo "       carry a trailing period on EVERY path, reference included. Rescore from"
echo "       the logged raw samples before comparing anything."
