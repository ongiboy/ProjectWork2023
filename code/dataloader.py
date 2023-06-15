import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from augmentations import DataTransform_FD, DataTransform_TD
import torch.fft as fft

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode, target_dataset_size=64, subset=False, enable_new_augs=False):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode
        X_train = dataset["samples"]
        y_train = dataset["labels"]
        # shuffle
        data = list(zip(X_train, y_train))
        np.random.shuffle(data)
        X_train, y_train = zip(*data)
        X_train, y_train = torch.stack(list(X_train), dim=0), torch.stack(list(y_train), dim=0)

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        """Align the TS length between source and target datasets"""
        X_train = X_train[:, :1, :int(config.TSlength_aligned)] # take the first 178 samples

        """Subset for debugging"""
        if subset == True:
            subset_size = target_dataset_size *10
            """if the dimension is larger than 178, take the first 178 dimensions. If multiple channels, take the first channel"""
            X_train = X_train[:subset_size] #
            y_train = y_train[:subset_size]

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        """Transfer x_data to Frequency Domain. If use fft.fft, the output has the same shape; if use fft.rfft, 
        the output shape is half of the time window."""

        window_length = self.x_data.shape[-1]
        self.x_data_f = fft.fft(self.x_data).abs() #/(window_length) # rfft for real value inputs.
        # self.x_data_f = self.x_data_f[:, :, 1:] # not a problem.

        self.len = X_train.shape[0]
        """Augmentation"""
        if training_mode == "pre_train":  # no need to apply Augmentations in other modes
            self.aug1 = DataTransform_TD(self.x_data, config, enable_new_augs=enable_new_augs)
            self.aug1_f = DataTransform_FD(self.x_data_f, config, enable_new_augs=enable_new_augs) # [7360, 1, 90]

    def __getitem__(self, index):
        if self.training_mode == "pre_train":
            return self.x_data[index], self.y_data[index], self.aug1[index],  \
                   self.x_data_f[index], self.aug1_f[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], \
                   self.x_data_f[index], self.x_data_f[index]

    def __len__(self):
        return self.len


def data_generator(sourcedata_path, targetdata_path, configs, training_mode, subset = True, enable_new_augs=False):

    train_dataset = torch.load(os.path.join(sourcedata_path, "train.pt"))
    pretrain_loss_dataset = torch.load(os.path.join(sourcedata_path, "test.pt"))
    finetune_dataset = torch.load(os.path.join(targetdata_path, "train.pt"))
    test_dataset = torch.load(os.path.join(targetdata_path, "test.pt"))
    """ Dataset notes:
    Epilepsy: train_dataset['samples'].shape = torch.Size([7360, 1, 178]); binary labels [7360] 
    valid: [1840, 1, 178]
    test: [2300, 1, 178]. In test set, 1835 are positive sampels, the positive rate is 0.7978"""
    """sleepEDF: finetune_dataset['samples']: [7786, 1, 3000]"""

    # subset = True # if true, use a subset for debugging.
    train_dataset = Load_Dataset(train_dataset, configs, training_mode, target_dataset_size=configs.batch_size, subset=subset, enable_new_augs=enable_new_augs) # for self-supervised, the data are augmented here
    pretrain_loss_dataset = Load_Dataset(pretrain_loss_dataset, configs, training_mode, target_dataset_size=configs.batch_size, subset=subset, enable_new_augs=enable_new_augs)
    finetune_dataset = Load_Dataset(finetune_dataset, configs, training_mode, target_dataset_size=configs.target_batch_size, subset=subset, enable_new_augs=enable_new_augs)
    if test_dataset['labels'].shape[0]>10*configs.target_batch_size:
        test_dataset = Load_Dataset(test_dataset, configs, training_mode, target_dataset_size=configs.target_batch_size*10, subset=subset, enable_new_augs=enable_new_augs)
    else:
        test_dataset = Load_Dataset(test_dataset, configs, training_mode, target_dataset_size=configs.target_batch_size, subset=subset, enable_new_augs=enable_new_augs)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)

    loss_loader = torch.utils.data.DataLoader(dataset=pretrain_loss_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)

    """the valid and test loader would be finetuning set and test set."""
    valid_loader = torch.utils.data.DataLoader(dataset=finetune_dataset, batch_size=configs.target_batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.target_batch_size,
                                              shuffle=True, drop_last=False,
                                              num_workers=0)

    return train_loader, loss_loader, valid_loader, test_loader