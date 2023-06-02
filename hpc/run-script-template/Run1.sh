#!/bin/bash
#BSUB -J Run1
#BSUB -o ../runs/Run1_%J.out
#BSUB -e ../runs/Run1_%J.err
#BSUB -q hpc
#BSUB -W 2
#BSUB -R "rusage[mem=512MB]"
#BSUP -n 4
#BSUB -N

# all  BSUB option comments should be above this line!

# execute our command
python ../../code/main.py --training_mode pre_train --pretrain_dataset SleepEEG --target_dataset Epileps --subset False
