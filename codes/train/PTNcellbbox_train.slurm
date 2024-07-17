#!/bin/bash
#SBATCH --job-name=query_PTNcellbbox_train
#SBATCH -o output/log.out.%j
#SBATCH -e output/log.err.%j
#SBATCH --partition=q01
#SBATCH --nodelist=g04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2 
#SBATCH --cpus-per-task=8 

CUDA_VISIBLE_DEVICES=0,1 mim train mmdet /share/home/pangliucheng/lc/project_table/config/cellwcls_queryinst_small_PTN_train.py --gpus 2 --no-validate --launcher pytorch
