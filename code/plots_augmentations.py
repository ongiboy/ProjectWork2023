import torch
import matplotlib.pyplot as plt
import numpy as np
import math
from augmentations import *
import torch.fft as fft
#remove .cuda in augmentations to plot !!!

sleep_data = torch.load('datasets\\SleepEEG\\train.pt')
x0 = sleep_data['samples'][:100]
x0_f = fft.fft(x0).abs()

x_reverse = reverse(x0)
x_flip = flip(x0)
x_spike = spike(x0, num_spikes= 8, max_spike=4)
x_slope = slope_trend(x0, max_stds=3)
x_step = step_trend(x0, num_steps=6, max_step=5)
x_noise = noise_replace(x0_f, 0.1)
x_scale = scale(x0_f)

# FREQUENCY DOMAIN OWN AUGMENTATION PLOTS_______________________________________________________________________________
x0_f_np = x0_f.numpy()
freq = np.fft.fftfreq(x0_f_np.shape[2])

x_noise_np = x_noise.numpy()
x_scale_np = x_scale.numpy()

# 3 plots, original fd, noise fd and scale fd.
fig, axs = plt.subplots(3, 1, sharex=True)

color = "darkslateblue"
titles = ['Original Signal (FD)', 'Noise Replace (FD)', 'Scale (FD)']

for i, data in enumerate([(x0_f_np, color), (x_noise_np, color), (x_scale_np, color)]):
    markerline, stemlines, baseline = axs[i].stem(freq, data[0][0][0], bottom=0,
                                                  linefmt='-', markerfmt='o', use_line_collection=True)
    markerline.set_markersize(3)
    markerline.set_color(data[1])
    stemlines.set_color(data[1])
    baseline.set_color("black")
    baseline.set_linewidth(1)
    axs[i].set_ylabel('Magnitude')
    axs[i].set_title(titles[i])

plt.xlabel('Frequency')
plt.show()

# TIME DOMAIN OWN AUGMENTATION PLOTS____________________________________________________________________________________
transformations = [x0, x_reverse, x_flip, x_spike, x_slope, x_step]
titles = ['Original Signal (TD)', 'Reverse (TD)', 'Flip horizontal (TD)', 'Spikes (TD)', 'Slope trend (TD)', 'Step trend (TD)']

num_plots = len(transformations)
fig, axs = plt.subplots(num_plots, 1)

for i in range(num_plots):
    axs[i].plot(transformations[i][0][0], linewidth=2)
    axs[i].set_title(titles[i])

fig.tight_layout()
plt.show()


# ARTICLE AUGMENTATION PLOTS____________________________________________________________________________________________
x_f = fft.fft(x0).abs()

aug_add = add_frequency(x_f, 0.3)
aug_remove = remove_frequency(x_f, 0.3)

## Using the jitter-augmentation
aug_jit = jitter(x0, 10)

## Using the scaling-augmentation
aug_scaling = scaling(x0, 3)

## Using the permutation-augmentation
#x_perm = torch.from_numpy(np.copy(x0))
aug_perm = permutation(x0, 6)

## Using the masking-augmentation
x_mask = torch.from_numpy(np.copy(x0))
aug_mask = masking(x_mask)

# OLD AUGMENTATIONS TD: original, perm, jitt, scaling, masking
fig, axs = plt.subplots(5)

axs[0].plot(x0[0][0], linewidth=2)
axs[0].set_title("Original Signal (TD)")

axs[1].plot(aug_perm[0][0], linewidth=2)
axs[1].set_title("Permutation (TD)")

axs[2].plot(aug_jit[0][0], linewidth=2)
axs[2].set_title("Jittering (TD)")

axs[3].plot(aug_scaling[0][0], linewidth=2)
axs[3].set_title("Scaling (TD)")

axs[4].plot(aug_mask[0][0], linewidth=2)
axs[4].set_title("Masking (TD)")

plt.tight_layout()
plt.show()

# OLD AUGMENTATIONS FD: original and add/remove freq
fig, axs = plt.subplots(3, 1, sharex=True)

color = "darkslateblue"
titles = ['Original Signal (FD)', 'Add frequency (FD)', 'Remove frequency (FD)']

freq = np.fft.fftfreq(x_f.shape[2])  # Frequency values for the frequency domain plots
for i, data in enumerate([(x_f, color), (aug_add, color), (aug_remove, color)]):
    markerline, stemlines, baseline = axs[i].stem(freq, data[0][0][0], bottom=0,
                                                  linefmt='-', markerfmt='o', use_line_collection=True)
    markerline.set_markersize(3)
    markerline.set_color(data[1])
    stemlines.set_color(data[1])
    baseline.set_color("black")
    baseline.set_linewidth(1)
    axs[i].set_ylabel('Magnitude')
    axs[i].set_title(titles[i])

plt.xlabel('Frequency')
plt.show()