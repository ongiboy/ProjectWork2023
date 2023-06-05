#!/bin/bash
#BSUB -J pre_Sleep_Epilepsy
#BSUB -o hpc/runs/Run1_%J.out
#BSUB -e hpc/runs/Run1_%J.err
#BSUB -q hpc
#BSUB -W 2
#BSUB -R "rusage[mem=512MB]"
#BSUB -n 4
#BSUB -N

# since all commands are from xterm's cd,
# remember to place cd in git folder: "ProjectWork2023"

python code/main.py --training_mode pre_train --pretrain_dataset SleepEEG --target_dataset Epilepsy --subset True