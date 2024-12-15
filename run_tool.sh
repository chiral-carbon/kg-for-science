#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --job-name=gradio-tool

# module --force purge
# module load modules/2.2-20230808 modules/2.3-20240529
# module load gcc/10.3.0 cuda/12.1.1 python/3.11.7

overlay=/scratch/ad6489/pyexample/overlay_img
img=/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif

singularity exec \
	--overlay $overlay:ro \
	$img \
       	/bin/bash -c \
	"source /ext3/env.sh; cd /scratch/ad6489/kg-for-science; conda activate kg4s_env; python scripts/run_db_interface_func.py"
