#!/bin/bash

#SBATCH --job-name=pm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p batch_grad
#SBATCH -w ariel-g4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=20G
#SBATCH --time=2-0
#SBATCH -o %N_%x_%j.out
#SBTACH -e %N_%x_%j.err

source /data/dlwogh9344/init.sh
conda activate torch38gpu
python main.py --dataset iDigits --num_tasks 4 --domain_inc --no_train_mask --epochs 1