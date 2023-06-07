#!/bin/bash
#BSUB -J Run_SleepEEG_Epilepsy
#BSUB -o hpc/runs/Run_virk_nu_%J.out
#BSUB -e hpc/runs/Run_virk_nu_%J.err

# gpu
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"

# runtime
#BSUB -W 4:00

# specs
#BSUB -R "rusage[mem=5GB] span[hosts=1]"
#BSUB -n 4

# mail when done
#BSUB -N

source myenv/bin/activate

python code/main.py --training_mode pre_train --pretrain_dataset SleepEEG --target_dataset Epilepsy --subset False

# python code/main.py --training_mode pre_train --pretrain_dataset ECG --target_dataset EMG --subset False

# python code/main.py --training_mode pre_train --pretrain_dataset HAR --target_dataset Gesture --subset False