import torch
import matplotlib.pyplot as plt
import numpy as np
import math
from augmentations import *
import torch.fft as fft

sleep_data = torch.load('datasets\\SleepEEG\\train.pt')
x0 = sleep_data['samples'][:100]
print(x0.shape)

x_reverse = reverse(x0)
x_flip = flip(x0)
x_spike = spike(x0, num_spikes=x0.shape[2]/10)
x_slope = slope_trend(x0, max_stds=2)
x_step = step_trend(x0, num_steps=2, max_step=x0.shape[2]/20)

jit_fig, ax = plt.subplots(6)

ax[0].plot(x0[0][0], linewidth=1)
ax[0].set_title("Original Signal (TD)")

ax[1].plot(x_reverse[0][0], linewidth=1)
ax[1].set_title("Reverse (TD)")

ax[2].plot(x_flip[0][0], linewidth=1)
ax[2].set_title("Flip horizontal (TD)")

ax[3].plot(x_spike[0][0], linewidth=1)
ax[3].set_title("Spikes (TD)")

ax[4].plot(x_slope[0][0], linewidth=1)
ax[4].set_title("Slope trend (TD)")

ax[5].plot(x_step[0][0], linewidth=1)
ax[5].set_title("Step trend (TD)")

jit_fig.tight_layout()
plt.show()
'''
x_f = fft.fft(x0).abs()

aug_add = add_frequency(x_f,0.2)
aug_remove = remove_frequency(x_f,0.5)
'''
'''fig, ax = plt.subplots(3)
ax[0].plot(x_f[0][0], linewidth=3)
ax[0].set_title("Original Signal")
ax[1].plot(aug_add, color = "darkslateblue", linewidth=3)
ax[1].set_title("Add frequency")
ax[2].plot(aug_remove, linewidth=3)
ax[2].set_title("Remove frequency")
fig.tight_layout()
plt.show()'''
'''
## Using the jitter-augmentation
aug_jit = jitter(x0,10)

## Using the scaling-augmentation
aug_scaling = scaling(x0,2)

## Using the permutation-augmentation
#x_perm = torch.from_numpy(np.copy(x0))
aug_perm = permutation(x0, 5)

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
ax[6].plot(aug_add[0][0], color = "darkslateblue", linewidth=2)
ax[6].set_title("Add frequency (FD)")
ax[7].plot(aug_remove[0][0], color = "darkslateblue", linewidth=2)
ax[7].set_title("Remove frequency (FD)")
jit_fig.tight_layout()
plt.show()
'''
