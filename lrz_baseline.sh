#!/bin/bash
#SBATCH -J medklip_pretrain
#SBATCH -N 1
#SBATCH -p mcml-hgx-a100-80x4
#SBATCH -q mcml
#SBATCH --mem=64GB
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH -o %x.%j.%N.out
#SBATCH -e %x.%j.%N.err


source ~/.bashrc  # activate miniconda
source ~/miniconda3/bin/activate medklip # activate your environment

# cd ~/MedKLIP
# cd PreTrain_MedKLIP/models/ops/
# sh ./make.sh
# cd -

# srun python test.py

python PreTrain_MedKLIP/train_MedKLIP.py --config PreTrain_MedKLIP/configs/baseline_lrz.yaml --output_dir output/pretrain/baseline
# python Sample_Finetuning_SIIMACR/I1_classification/train_medklip.py --config Sample_Finetuning_SIIMACR/I1_classification/configs/baseline.yaml --output_dir output/baseline --pretrain_path output/pretrain/baseline