#!/usr/bin/env bash

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7}"

L4GM_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export L4GM_REPO_ROOT="${L4GM_REPO_ROOT:-$(cd "${L4GM_SCRIPT_DIR}/.." && pwd)}"
export L4GM_CACHE_ROOT="${L4GM_CACHE_ROOT:-${L4GM_REPO_ROOT}/.cache}"
mkdir -p \
  "${L4GM_CACHE_ROOT}/huggingface" \
  "${L4GM_CACHE_ROOT}/torch" \
  "${L4GM_CACHE_ROOT}/xdg" \
  "${L4GM_CACHE_ROOT}/pip"

export HF_HOME="${HF_HOME:-${L4GM_CACHE_ROOT}/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${L4GM_CACHE_ROOT}/huggingface}"
export DIFFUSERS_CACHE="${DIFFUSERS_CACHE:-${L4GM_CACHE_ROOT}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${L4GM_CACHE_ROOT}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${L4GM_CACHE_ROOT}/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${L4GM_CACHE_ROOT}/pip}"
export IMAGEDREAM_MODEL_PATH="${IMAGEDREAM_MODEL_PATH:-${L4GM_REPO_ROOT}/pretrained/imagedream-ipmv-diffusers}"

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

export PYTHONPATH="${L4GM_REPO_ROOT}:${PYTHONPATH:-}"
