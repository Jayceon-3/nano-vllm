#!/usr/bin/env bash
# This file is sourced by an interactive shell, so preserve its shell options.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || return
module load python3.11-anaconda/2024.02 || return
source "$(conda info --base)/etc/profile.d/conda.sh" || return
conda activate nanovllm
