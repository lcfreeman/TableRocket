#!/bin/bash
#SBATCH --job-name=query_FTNcellbbox_val
#SBATCH -o output/log.out.%j
#SBATCH -e output/log.err.%j
#SBATCH --partition=q01
#SBATCH --nodelist=g02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2 
#SBATCH --cpus-per-task=8 


CUDA_VISIBLE_DEVICES=0,1 mim test mmdet /share/home/pangliucheng/lc/project_table/config/cellwcls_queryinst_small_FTN_train.py --gpus 2 --checkpoint work_dirs/cellwlcs_queryinst_small_FTN_train/epoch_23.pth --show --launcher pytorch --eval bbox
