import torch
import matplotlib.pyplot as plt
import numpy as np
import math
from augmentations import *
import torch.fft as fft
from augmentations import DataTransform_FD, DataTransform_TD


sleep_data = torch.load("C:/Users/danie/TFC-pretraining/datasets/SleepEEG/train.pt")
x0 = sleep_data['samples']
x0.shape

x_f = fft.fft(x0).abs()

aug_add = add_frequency(x_f[0][0],0.2)
aug_remove = remove_frequency(x_f[0][0],0.5)

'''fig, ax = plt.subplots(3)
ax[0].plot(x_f[0][0], linewidth=3)
ax[0].set_title("Original Signal")
ax[1].plot(aug_add, color = "darkslateblue", linewidth=3)
ax[1].set_title("Add frequency")
ax[2].plot(aug_remove, linewidth=3)
ax[2].set_title("Remove frequency")
fig.tight_layout()
plt.show()'''



## Using the jitter-augmentation
aug_jit = jitter(x0,10)

## Using the scaling-augmentation
aug_scaling = scaling(x0,2)

## Using the permutation-augmentation
x_perm = torch.from_numpy(np.copy(x0))
aug_perm = permutation(x_perm, 5)

## Using the masking-augmentation
x_mask = torch.from_numpy(np.copy(x0))
aug_mask = masking(x_mask, 0.4)


jit_fig, ax = plt.subplots(8)
ax[0].plot(x0[0][0], linewidth=2)
ax[0].set_title("Original Signal (TD)")
ax[1].plot(aug_perm[0][0], linewidth=2)
ax[1].set_title("Permutation (TD)")
ax[2].plot(aug_jit[0][0], linewidth=2)
ax[2].set_title("Jittering (TD)")
ax[3].plot(aug_scaling[0][0], linewidth=2)
ax[3].set_title("Scaling (TD)")
ax[4].plot(aug_mask[0][0], linewidth=2)
ax[4].set_title("Masking (TD)")
ax[5].plot(x_f[0][0], color = "darkslateblue", linewidth=2)
ax[5].set_title("Original Signal (FD)")
ax[6].plot(aug_add, color = "darkslateblue", linewidth=2)
ax[6].set_title("Add frequency (FD)")
ax[7].plot(aug_remove, color = "darkslateblue", linewidth=2)
ax[7].set_title("Remove frequency (FD)")
jit_fig.tight_layout()
plt.show()

