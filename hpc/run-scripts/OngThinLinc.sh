#!/bin/bash
#BSUB -J Run3_pre_Sleep_Epilepsy
#BSUB -o hpc/runs/Run3_%J.out
#BSUB -e hpc/runs/Run3_%J.err

# gpu
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 02:00
#BSUB -R "rusage[mem=8G] span[hosts=1]"
#BSUB -n 4
#BSUB -N

# since all commands are from xterm's cd,
# remember to place cd in git folder: "ProjectWork2023"

source environments/PW_env/bin/activate

python code/main.py --training_mode pre_train --pretrain_dataset SleepEEG --target_dataset Epilepsy --subset True