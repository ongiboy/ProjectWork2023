import torch
import numpy as np
import matplotlib.pyplot as plt

#            "HAR", #6 "Gesture", and EXO another type: DOUBLE TENSORS
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

#CLASS DISTRIBUTION IN %   (HAR, Gesture and Exo not included, because "bincount_cpu" not implemented for 'Double')
for i in range(len(datasets)):
    if datasets[i] != datasets[6] and datasets[i] != datasets[7] and datasets[i] != datasets[-1]:
        if len(load[i]) == 3:
            all_labels = torch.cat([load[i][0]["labels"], load[i][1]["labels"], load[i][2]["labels"]])
        else:
            all_labels = torch.cat([load[i][0]["labels"], load[i][1]["labels"]])
        class_counts = torch.bincount(all_labels)
        total_samples = len(all_labels)
        class_distribution = class_counts.float() / total_samples * 100
        print(datasets[i], class_distribution)

# Depression signal plot of condition and control_______________________________________________________________________
Dep_cont_sig = (load[9][0]["samples"][5].view(5120), load[9][0]["labels"][5])
Dep_cond_sig = (load[9][0]["samples"][10].view(5120), load[9][0]["labels"][10])
fig, axs = plt.subplots(2, 1, figsize=(8, 8))

axs[0].plot(Dep_cont_sig[0][:1441], color='dodgerblue')
axs[0].set_title("Control signal")
axs[0].set_ylabel("Activity count")

axs[1].plot(Dep_cond_sig[0][:1441], color='tomato')
axs[1].set_title("Condition signal")
axs[1].set_ylabel("Activity count")

plt.tight_layout()
plt.show()

print(Dep_cont_sig[1])
print(Dep_cond_sig[1])

# Exoplanets signal plot of non- and Exoplanet__________________________________________________________________________
Exo_False_sig = (load[-1][0]["samples"][0].view(5120), load[-1][0]["labels"][0])
Exo_True_sig = (load[-1][0]["samples"][40].view(5120), load[-1][0]["labels"][40])

fig, axs = plt.subplots(2, 1, figsize=(8, 8))

axs[0].plot(Exo_False_sig[0][:3198], color='darkblue')
axs[0].set_title("No Exoplanet signal")
axs[0].set_ylabel("Flux")

axs[1].plot(Exo_True_sig[0][:3198], color='darkgreen')
axs[1].set_title("Exoplanet signal")
axs[1].set_ylabel("Flux")

plt.tight_layout()
plt.show()

print(Exo_False_sig[1])
print(Exo_True_sig[1])