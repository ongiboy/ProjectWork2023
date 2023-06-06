#!/bin/bash
#BSUB -J Run_SleepEEG_Epilepsy
#BSUB -o hpc/runs/Run_SleepEEG_Epilepsy_%J.out
#BSUB -e hpc/runs/Run_SleepEEG_Epilepsy_%J.err

# gpu
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 20:00
#BSUB -R "rusage[mem=100G] span[hosts=1]"
#BSUB -n 32
#BSUB -N
# excute or command

source ../myenv/env/bin/activate

python code/main.py --training_mode pre_train --pretrain_dataset SleepEEG --target_dataset Epilepsy --subset False