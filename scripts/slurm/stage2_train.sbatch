#!/bin/bash
#SBATCH --job-name=raft_stage2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --mem=384G
#SBATCH --time=48:00:00
# Example Slurm wrapper; adjust partition/account for your cluster.
#   sbatch --export=ALL,INIT_CKPT_DIR=...,EXP_NAME=... scripts/slurm/stage2_train.sbatch
set -euo pipefail
bash "${RAFT_HOME:-$SLURM_SUBMIT_DIR}/scripts/stage2/train_action.sh"
