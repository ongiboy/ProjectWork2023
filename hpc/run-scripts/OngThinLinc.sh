#!/bin/bash
#BSUB -J pre_HAR_Gesture_subYes
#BSUB -o hpc/runs/Run_%J.out.txt
#BSUB -e hpc/runs/Run_%J.err.txt

# gpu
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"

# runtime
#BSUB -W 20:00

# specs
#BSUB -R "rusage[mem=8G] span[hosts=1]"
#BSUB -n 4

# mail when done
#BSUB -N

# since all commands are from xterm's cd,
# remember to place xterm cd in git folder: "ProjectWork2023"

source hpc/environments/PW_env/bin/activate

python code/main.py --training_mode pre_train --pretrain_dataset HAR --target_dataset Gesture --subset True
# SleepEEG -> Epilepsy
# FD_A -> FD_B
# ECG -> EMG
# HAR -> Gesture