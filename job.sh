#!/bin/bash
#
#SBATCH --job-name=mixture-of-tokens-team-ml
#SBATCH --partition=common
#SBATCH --qos=1gpu1d
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=mixture-of-tokens-team-ml.out
#SBATCH --error=mixture-of-tokens-team-ml.err


python train.py --experiment-name=really_small_c4
# python train.py --experiment-name=mot_paper_small_c4
# python train.py --experiment-name=mot_medium_32_8
