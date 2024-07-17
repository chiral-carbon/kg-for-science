#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --constraint=icelake,ib
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=2
#SBATCH -C a100-80gb&ib-a100
#SBATCH --mem=512G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --job-name=kg-for-science

module --force purge
module load modules/2.2-20230808 modules/2.3-20240529
module load gcc/10.3.0 cuda/12.1.1 python/3.11.7
echo "$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | sort | uniq -c | awk -v hostname="$(hostname)" \
     '{print "Running on " hostname " with", $1, $2, $3, $4, "GPUs."}')"

cd /mnt/home/adas1/projects/knowledge-graph/kg-for-science
conda init
conda activate kg4s
python main.py --sweep