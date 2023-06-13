import torch
import numpy as np

datasets = ["SleepEEG", #0
            "Epilepsy", 

            "FD_A", #2
            "FD_B",

            "ECG", #4
            "EMG",

            "HAR", #6
            "Gesture",
            
            "Depression_SleepEEG", #8
            "Depression_FD_A",
            "Depression_HAR", #10
            "Depression_ECG"
            ]
load = np.array([])
for data in datasets:
    train = torch.load(f'datasets\\{data}\\train.pt')
    val = torch.load(f'datasets\\{data}\\val.pt')
    test = torch.load(f'datasets\\{data}\\test.pt')
    temp = np.array([train, val, test])

    if load.shape == (0,):
        load = temp
    else:
        load = np.vstack((load,temp))

shapes = np.array([])
for i in range(load.shape[0]):
    temp = np.array([[load[i][0]["samples"].shape, load[i][1]["samples"].shape, load[i][2]["samples"].shape]])

    if shapes.shape == (0,):
        shapes = temp
    else:
        shapes = np.vstack((shapes,temp))

# shapes[dataset][train/val/test] = shape
print()