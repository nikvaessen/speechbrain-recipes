#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --account=nvaessen
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem 20G
#SBATCH --gres=gpu:1
#SBATCH -o results/slurm/%j.out
#SBATCH -e results/slurm/%j.err
#SBATCH --mail-user=nvaessen
#SBATCH --mail-type=BEGIN,END,FAIL

poetry run python train_speaker_embeddings.py hparams/train_x_vectors.yaml
