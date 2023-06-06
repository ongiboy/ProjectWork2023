#!/bin/bash
#BSUB -J pre_Sleep_Epilepsy_subNot
#BSUB -o hpc/runs/%J.out
#BSUB -e hpc/runs/J.err

# gpu
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"

# runtime
#BSUB -W 20:00

# specs
#BSUB -R "rusage[mem=16G] span[hosts=1]"
#BSUB -n 4

# mail when done
#BSUB -N

# since all commands are from xterm's cd,
# remember to place xterm cd in git folder: "ProjectWork2023"

source hpc/environments/PW_env/bin/activate

python code/main.py --training_mode pre_train --pretrain_dataset SleepEEG --target_dataset Epilepsy --subset False
# SleepEEG -> Epilepsy
# FD-A -> FD-B
# ECG -> EMG
# HAR -> Gesture