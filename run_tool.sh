#!/bin/bash
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --job-name=gradio-tool

module --force purge
module load modules/2.2-20230808 modules/2.3-20240529
module load gcc/10.3.0 cuda/12.1.1 python/3.11.7

cd /mnt/home/adas1/projects/knowledge-graph/kg-for-science
conda init
conda activate kg4s
python -u app.py