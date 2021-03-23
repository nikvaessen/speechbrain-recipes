#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --account=nvaessen
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem 20G
#SBATCH --gres=gpu:1
#SBATCH -o /scratch-csedu/other/nik/slurm/%j.out
#SBATCH -e /scratch-csedu/other/nik/slurm/%j.err
#SBATCH --mail-user=nvaessen
#SBATCH --mail-type=BEGIN,END,FAIL

poetry run bash run_experiment.sh "$1"
