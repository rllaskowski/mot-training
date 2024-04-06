#!/bin/bash
#
#SBATCH --job-name=mixture-of-tokens-team-ml
#SBATCH --partition=common
#SBATCH --qos=8gpu15m
#SBATCH --gres=gpu:2
#SBATCH --time=15
#SBATCH --output=mixture-of-tokens-team-ml.out

python train.py
