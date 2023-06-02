#!/bin/bash
#BSUB -J hpc_run
#BSUB -o ../run-scripts-ignore%J.out
#BSUB -e ../run-scripts-ignore%J.err
#BSUB -q hpc
#BSUB -W 2
#BSUB -R "rusage[mem=512MB]"
#BSUB -u christianonghansen@gmail.com
#BSUB -N
# excute or command

python python code\main.py --training_mode fine_tune_test --pretrain_dataset SleepEEG --target_dataset Epilepsy