#!/bin/bash
#BSUB -J hpc_run
#BSUB -o hpc/runs%J.out
#BSUB -e hpc/runs%J.err
#BSUB -q gpuv100
#BSUB -W 01:00
#BSUB -R "rusage[mem=512MB]"
#BSUB -u christianonghansen@gmail.com
#BSUB -n 32
#BSUB -N
# excute or command

source ../myenv/env/bin/activate

python code/main.py --training_mode fine_tune_test --pretrain_dataset SleepEEG --target_dataset Epilepsy