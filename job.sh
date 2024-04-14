#!/bin/bash
#
#SBATCH --job-name=mixture-of-tokens-team-ml
#SBATCH --partition=common
#SBATCH --qos=8gpu15m
#SBATCH --gres=gpu:2
#SBATCH --time=6:00
#SBATCH --output=mixture-of-tokens-team-ml.out
#SBATCH --error=mixture-of-tokens-team-ml.err
#SBATCH --node-list=asusgpu1

python train.py
