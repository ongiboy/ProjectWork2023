#!/bin/bash
#BSUB -J pre_Sleep_Epi_subYes
#BSUB -o hpc/runs/Run_%J.out.txt
#BSUB -e hpc/runs/Run_%J.err.txt

# gpu
#BSUB -q gpua10
#BSUB -gpu "num=1:mode=exclusive_process"

# runtime
#BSUB -W 6:00

# specs
#BSUB -R "rusage[mem=10GB] span[hosts=1]"
#BSUB -n 4

# mail when done
#BSUB -N

# since all commands are from xterm's cd,
# remember to place xterm cd in git folder: "ProjectWork2023"

source hpc/environments/PW_env/bin/activate

# SleepEEG -> Epilepsy
python code/main.py --training_mode pre_train --pretrain_dataset SleepEEG --target_dataset Epilepsy --subset True
# FD_A -> FD_B
# python code/main.py --training_mode pre_train --pretrain_dataset FD-A --target_dataset FD-B --subset False
# ECG -> EMG
# python code/main.py --training_mode pre_train --pretrain_dataset ECG --target_dataset EMG --subset False
# HAR -> Gesture
# python code/main.py --training_mode pre_train --pretrain_dataset HAR --target_dataset Gesture --subset False
