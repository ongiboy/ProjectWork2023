import torch
import numpy as np

#            "HAR", #6 "Gesture", another type: DOUBLE TENSORS
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
            "Depression_ECG",
            "Exoplanets_FD_A" #12
            ]

load = np.array([])
for data in datasets:
    train = torch.load(f'datasets\\{data}\\train.pt')
    test = torch.load(f'datasets\\{data}\\test.pt')

    if data in datasets[:8]:
        val = torch.load(f'datasets\\{data}\\val.pt')

        temp = np.array([train, val, test])
    temp = np.array([train, test])

    if load.shape == (0,):
        load = temp
    else:
        load = np.vstack((load,temp))

shapes = np.array([])
for i in range(load.shape[0]):
    if len(load[i]) == 3:
        temp = np.array([[load[i][0]["samples"].shape, load[i][1]["samples"].shape, load[i][2]["samples"].shape]])
    else:
        temp = np.array([[load[i][0]["samples"].shape, load[i][1]["samples"].shape]])

    if shapes.shape == (0,):
        shapes = temp
    else:
        shapes = np.vstack((shapes,temp))

# shapes[dataset][train/val/test] = shape

# CLASS DISTRIBUTION IN %   (HAR and Gesture not included, because "bincount_cpu" not implemented for 'Double')
for i in range(len(datasets)):
    if datasets[i] != datasets[6] and datasets[i] != datasets[7]:
        if len(load[i]) == 3:
            all_labels = torch.cat([load[i][0]["labels"], load[i][1]["labels"], load[i][2]["labels"]])
        else:
            all_labels = torch.cat([load[i][0]["labels"], load[i][1]["labels"]])
        class_counts = torch.bincount(all_labels)
        total_samples = len(all_labels)
        class_distribution = class_counts.float() / total_samples * 100
        print(datasets[i], class_distribution)