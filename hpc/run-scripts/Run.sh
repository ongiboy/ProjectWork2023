#!/bin/bash
#BSUB -J Sl_Ep_p
## ps (pretrain-subset), f (tinetune-nosubset)
#BSUB -o hpc/runs/Run_%J.out.txt
#BSUB -e hpc/runs/Run_%J.err.txt

## GPU
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"

## runtime
#BSUB -W 3:00

## specs
#BSUB -R "rusage[mem=12GB] span[hosts=1]"
#BSUB -n 6

## mail when done
#BSUB -N

## since all commands are from xterm's cd,
## remember to place xterm cd in git folder: "ProjectWork2023"

module load python3/3.9.11

source ../HPC_env/bin/activate

## SleepEEG -> Epilepsy (n2: 2hr)
python code/main.py --training_mode pre_train --run_description Run1 --pretrain_dataset SleepEEG --target_dataset Epilepsy --subset False --device cuda
## FD_A -> FD_B
# python code/main.py --training_mode fine_tune_test --pretrain_dataset FD_A --target_dataset FD_B --subset False --device cuda
## ECG -> EMG (>8GB pre)
# python code/main.py --training_mode pre_train --pretrain_dataset ECG --target_dataset EMG --subset False --device cuda
## HAR -> Gesture (pre_n2: 5hr)
#python code/main.py --training_mode pre_train --pretrain_dataset HAR --target_dataset Gesture --subset False --device cuda
