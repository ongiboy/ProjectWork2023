import numpy as np
import torch
import random
import matplotlib.pyplot as plt
#import tensorflow as tf

def one_hot_encoding(X):
    X = [int(x) for x in X]
    n_values = np.max(X) + 1
    b = np.eye(n_values)[X]
    return b

def DataTransform(sample, config):
    """Weak and strong augmentations"""
    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    # weak_aug = permutation(sample, max_segments=config.augmentation.max_seg)
    strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)

    return weak_aug, strong_aug

# def DataTransform_TD(sample, config):
#     """Weak and strong augmentations"""
#     weak_aug = sample
#     strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio) #masking(sample)
#     return weak_aug, strong_aug
#
# def DataTransform_FD(sample, config):
#     """Weak and strong augmentations in Frequency domain """
#     # weak_aug =  remove_frequency(sample, 0.1)
#     strong_aug = add_frequency(sample, 0.1)
#     return weak_aug, strong_aug
def DataTransform_TD(sample, config):
    if False or config.aug_new != "": # OBS
        print("New augmentations in use")
        if config.aug_new == "Depr": #
            pass
        elif config.aug_new == "Exo": #
            pass
    else:
        print("Article augmentation in use")
        """Weak and strong augmentations"""
        x_mask = torch.from_numpy(np.copy(sample))
        aug_1 = jitter(sample, config.augmentation.jitter_ratio)
        aug_2 = scaling(sample, config.augmentation.jitter_scale_ratio)
        aug_3 = permutation(sample, max_segments=config.augmentation.max_seg)
        aug_4 = masking(x_mask, keepratio=0.9)
        #plt.plot(sample[0][0],color="red")
        #plt.plot(aug_4[0][0],color="blue")

        li = np.random.randint(0, 4, size=[sample.shape[0]]) # there are two augmentations in Frequency domain
        li_onehot = one_hot_encoding(li)
        aug_1[li_onehot[:, 0]==0] = 0 # the rows are not selected are set as zero.
        aug_2[li_onehot[:, 1]==0] = 0
        aug_3[li_onehot[:, 2]==0] = 0
        aug_4[li_onehot[:, 3]==0] = 0
        aug_T = aug_1 + aug_2 + aug_3 + aug_4
        #plt.plot(aug_T[3][0])
    
    return aug_T


def DataTransform_FD(sample, config):
    if False or config.aug_new != "": # OBS
        print("New augmentations in use")
        if config.aug_new == "Depr": # reverse, noise injection,
            pass
        elif config.aug_new == "Exo": # flip, noise injection,
            pass
    else:
        """Weak and strong augmentations in Frequency domain """
        aug_1 =  remove_frequency(sample, 0.1)
        aug_2 = add_frequency(sample, 0.1)

        # generate random sequence
        li = np.random.randint(0, 2, size=[sample.shape[0]]) # there are two augmentations in Frequency domain
        li_onehot = one_hot_encoding(li)
        aug_1[li_onehot[:, 0]==0] = 0 # the rows are not selected are set as zero.
        aug_2[li_onehot[:, 1]==0] = 0
        aug_F = aug_1 + aug_2

    return aug_F



def generate_binomial_mask(B, T, D, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T, D))).to(torch.bool)

def masking(x, keepratio=0.9, mask= 'binomial'):
    global mask_id
    nan_mask = ~x.isnan().any(axis=-1)
    x[~nan_mask] = 0
    # x = self.input_fc(x)  # B x T x Ch

    if mask == 'binomial':
        mask_id = generate_binomial_mask(x.size(0), x.size(1), x.size(2), p=keepratio).to(x.device)
    # elif mask == 'continuous':
    #     mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
    # elif mask == 'all_true':
    #     mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
    # elif mask == 'all_false':
    #     mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
    # elif mask == 'mask_last':
    #     mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
    #     mask[:, -1] = False

    # mask &= nan_mask
    x[~mask_id] = 0
    return x

def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return torch.from_numpy(np.concatenate((ai), axis=1))

def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            random.shuffle(splits)
            warp = np.concatenate(splits).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

def remove_frequency(x, maskout_ratio=0):
    mask = torch.FloatTensor(x.shape).uniform_() > maskout_ratio # maskout_ratio are False
    mask = mask.to(x.device)
    return x*mask

def add_frequency(x, pertub_ratio=0):
    mask = torch.FloatTensor(x.shape).uniform_() > (1-pertub_ratio) # only pertub_ratio of all values are True
    mask = mask.to(x.device)
    max_amplitudes = torch.max(x, axis=2, keepdim=True)[0]
    max_amplitudes = max_amplitudes.repeat(1, 1, x.shape[2])
    #max_amplitude = x.max()
    random_am = torch.rand(mask.shape)*(max_amplitudes*0.5)
    pertub_matrix = mask*random_am
    return x+pertub_matrix


### OWN AUGMENTATIONS

# TD
def reverse():
    pass

def flip():
    pass

def spike():
    pass

def slope_trend():
    pass

def step_trend():
    pass

# FD
def noise_replace():
    pass