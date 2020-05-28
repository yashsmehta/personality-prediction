#!/bin/bash
#SBATCH --job-name=np
#SBATCH --output=tmp.out
#SBATCH --time=0-12:00
#SBATCH --nodes=1
#SBATCH --mem=1G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

srun -u python process_data.py