#!/bin/bash
#BSUB -J ProjectWorkTest
#BSUB -o ProjectWorkTest_%J.out
#BSUB -e ProjectWorkTest_%J.err
#BSUB -q hpc
#BSUB -W 2
#BSUB -R "rusage[mem=512MB]"
#BSUB -u christianonghansen@gmail.com
#BSUB -N
# excute or command

python python code\main.py --training_mode pre_train --pretrain_dataset SleepEEG --target_dataset Epilepsy