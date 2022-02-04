#!/bin/bash
#SBATCH -t 11:55:00
#SBATCH -c 1
#SBATCH --mem-per-cpu 8G

#SBATCH --job-name=train-fr
#SBATCH --output slurm_log/fr-%j.log

#module load anaconda

echo $(date)

srun python -u train-fr.py \
    --exp_name ${1} --subj_id ${2} --n ${3}  --n_std ${4} --dim_hidden ${5} \
    --lr ${6} --n_epochs ${7} --penalize_repeat ${8}

echo $(date)
