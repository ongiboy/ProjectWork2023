import numpy as np
import torch

def one_hot_encoding(X):
    X = [int(x) for x in X]
    n_values = np.max(X) + 1
    b = np.eye(n_values)[X]
    return b

def DataTransform_TD(sample, config, enable_new_augs=False):
    if enable_new_augs and config.aug_new != "":
        print("New augmentations in use")
        if config.aug_new == "Depr": # TD augments for depression fine-tuning
            aug_0 = masking(sample.clone())
            aug_1 = jitter(sample.clone(), config.augmentation.jitter_ratio)
            aug_2 = scaling(sample.clone(), config.augmentation.jitter_scale_ratio)
            aug_3 = spike(sample.clone(), num_spikes=len(sample)//50, max_spike=2)

            li = np.random.randint(0, 4, size=[sample.shape[0]]) # there are two augmentations in Frequency domain
            li_onehot = one_hot_encoding(li) == 1
            aug_0[~li_onehot[:, 0]] = 0
            aug_1[~li_onehot[:, 1]] = 0 # the rows are not selected (false) are set to zero.
            aug_2[~li_onehot[:, 2]] = 0
            aug_3[~li_onehot[:, 3]] = 0
            
            aug_T = aug_0 + aug_1 + aug_2 + aug_3

        elif config.aug_new == "Exo": # TD augments for Exoplanet finetuning
            aug_0 = masking(sample.clone())
            aug_1 = jitter(sample.clone(), config.augmentation.jitter_ratio)
            aug_2 = scaling(sample.clone(), config.augmentation.jitter_scale_ratio)
            aug_3 = reverse(sample.clone())
            aug_4 = flip(sample.clone())
            aug_5 = spike(sample.clone(), num_spikes=len(sample)//50, max_spike=2)
            aug_6 = slope_trend(sample.clone(), max_stds=1)
            aug_7 = step_trend(sample.clone(), num_steps=len(sample)//50, max_step=0.5)

            li = np.random.randint(0, 8, size=[sample.shape[0]]) # there are two augmentations in Frequency domain
            li_onehot = one_hot_encoding(li) == 1
            aug_0[~li_onehot[:, 0]] = 0
            aug_1[~li_onehot[:, 1]] = 0 # the rows are not selected (false) are set to zero.
            aug_2[~li_onehot[:, 2]] = 0
            aug_3[~li_onehot[:, 3]] = 0
            aug_4[~li_onehot[:, 4]] = 0
            aug_5[~li_onehot[:, 5]] = 0 # the rows are not selected (false) are set to zero.
            aug_6[~li_onehot[:, 6]] = 0
            aug_7[~li_onehot[:, 7]] = 0

            aug_T = aug_0 + aug_1 + aug_2 + aug_3 + aug_4 + aug_5 + aug_6 + aug_7

    else:
        print("Article augmentation in use")
        aug_1 = jitter(sample, config.augmentation.jitter_ratio)
        aug_2 = scaling(sample, config.augmentation.jitter_scale_ratio)
        aug_3 = permutation(sample, max_segments=config.augmentation.max_seg)

        li = np.random.randint(0, 3, size=[sample.shape[0]]) # there are two augmentations in Frequency domain
        li_onehot = one_hot_encoding(li) == 1
        aug_1[~li_onehot[:, 0]] = 0 # the rows are not selected (false) are set to zero.
        aug_2[~li_onehot[:, 1]] = 0
        aug_3[~li_onehot[:, 2]] = 0
        # aug_4[1 - li_onehot[:, 3]] = 0
        aug_T = aug_1 + aug_2 + aug_3 #+aug_4
    
    return aug_T


def DataTransform_FD(sample, config, enable_new_augs=False):
    if enable_new_augs and config.aug_new != "":
        print("New augmentations in use")
        if config.aug_new == "Depr": # reverse, noise injection,
            aug_0 = noise_replace(sample.clone(), ratio=0.05)
            aug_1 = scale(sample.clone())
            
        elif config.aug_new == "Exo": # flip, noise injection,
            aug_0 = noise_replace(sample.clone(), ratio=0.05)
            aug_1 = scale(sample.clone())

        # generate random sequence
        li = np.random.randint(0, 2, size=[sample.shape[0]]) # there are two augmentations in Frequency domain
        li_onehot = one_hot_encoding(li) == 1
        aug_0[~li_onehot[:, 0]] = 0 # the rows are not selected (false) are set to zero.
        aug_1[~li_onehot[:, 1]] = 0
        aug_F = aug_0 + aug_1

    else:
        """Weak and strong augmentations in Frequency domain """
        aug_1 =  remove_frequency(sample, 0.1)
        aug_2 = add_frequency(sample, 0.1)
        # generate random sequence
        li = np.random.randint(0, 2, size=[sample.shape[0]]) # there are two augmentations in Frequency domain
        li_onehot = one_hot_encoding(li) == 1
        aug_1[~li_onehot[:, 0]] = 0 # the rows are not selected (false) are set to zero.
        aug_2[~li_onehot[:, 1]] = 0
        aug_F = aug_1 + aug_2

    return aug_F



def generate_binomial_mask(B, T, D, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T, D))).to(torch.bool)

def masking(x, mask= 'binomial'):
    nan_mask = ~x.isnan().any(axis=-1)
    x[~nan_mask] = 0
    if mask == 'binomial':
        mask_id = generate_binomial_mask(x.size(0), x.size(1), x.size(2), p=0.9).to(x.device)
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
    return np.concatenate((ai), axis=1)

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
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

def remove_frequency(x, maskout_ratio=0):
    mask = torch.cuda.FloatTensor(x.shape).uniform_() > maskout_ratio # maskout_ratio are False
    mask = mask.to(x.device)
    return x*mask

def add_frequency(x, pertub_ratio=0,):

    mask = torch.cuda.FloatTensor(x.shape).uniform_() > (1-pertub_ratio) # only pertub_ratio of all values are True
    mask = mask.to(x.device)
    max_amplitude = x.max()
    random_am = torch.rand(mask.shape)*(max_amplitude*0.1)
    pertub_matrix = mask*random_am
    return x+pertub_matrix


### OWN AUGMENTATIONS

# TD
def reverse(x):
    flipped_x = torch.flip(x, dims=(2,))
    return flipped_x

def flip(x):
    return -x

def spike(x, num_spikes, max_spike=2):
    """adds random spikes at random index for each signal """
    spiked_x = x.numpy().copy()

    # loop over each signal/row
    for i in range(spiked_x.shape[0]):
        signal = spiked_x[i, 0, :]  # 0 if only 1 channel !!!
        std = np.std(signal)

        for _ in range(int(num_spikes)):
            # random generate spikes and augment data
            index = np.random.randint(0, len(signal))
            direction = np.random.choice([-1, 1])
            spike_magnitude = np.random.uniform(0, max_spike) * std

            spiked_x[i, 0, index] += direction * spike_magnitude  # add spike to signal
    return torch.from_numpy(spiked_x)

def slope_trend(x, max_stds=1):
    """adds a random slope to each signal"""
    sloped_x = x.numpy().copy()

    # loop over each signal/row
    for i in range(sloped_x.shape[0]):
        signal = sloped_x[i, 0, :]  # 0 if only 1 channel !!!
        std = np.std(signal)

        # generate random slope between 2 random points
        start_point = np.random.uniform(-1, 1) * std * max_stds
        end_point = np.random.uniform(-1, 1) * std * max_stds

        slope = (end_point-start_point)/len(signal)
        slope_array = np.array([j*slope for j in range(len(signal))])

        sloped_x[i, 0, :] += slope_array.astype('float64')
    return torch.from_numpy(sloped_x)

def step_trend(x, num_steps, max_step=1):
    """Adds step-like trends to signals"""
    stepped_x = x.numpy().copy()

    # loop over each signal/row
    for i in range(stepped_x.shape[0]):
        signal = stepped_x[i, 0, :]  # 0 if only 1 channel !!!
        std = np.std(signal)


        idxs = [np.random.randint(0, len(signal)+1) for _ in range(int(num_steps))]
        idxs.sort()
        points = [np.random.uniform(-max_step, max_step) * std for _ in range(int(num_steps))]

        step_array = np.zeros(len(signal))
        for i_step in range(int(num_steps)):
            if idxs[i_step] == idxs[0]:
                step_array[:idxs[i_step]+1] = points[i_step]
            else:
                step_array[idxs[i_step-1]:idxs[i_step]+1] = points[i_step]

        stepped_x[i, 0, :] += step_array
    return torch.from_numpy(stepped_x)

# FD
def noise_replace(x, ratio=0.05):
    """ Replaces random frequencies with random (uniform) values """
    # uniform random tensor - max values equal to max value of the corresponding row in x
    max_values, _ = torch.max(x, dim=2)
    max_tensor = max_values.unsqueeze(1).expand_as(x) 
    tensor_uniform = torch.rand_like(x) * max_tensor

    mask = torch.FloatTensor(x.shape).uniform_() > ratio # choose "ratio" amount of points
    mask = mask.to(x.device)
    x_muted = x*mask # removes certain points
    return x_muted + ~mask * tensor_uniform # the points removed are replaced by noise

def scale(x):
    """ Multiplies each signal in frequency domain with a scalar (same as scaling time domain) """
    #generates random numbers from a standard normal distribution, *0.2 + 1 scales and shifts the numbers.
    scals = torch.randn(x.shape[0], 1, 1) * 0.2 + 1.0
    return x * scals