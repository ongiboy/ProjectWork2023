import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def split_data(dataframe, training_mode):
    data = dataframe.values

    labels = data[:,0]-1 
    data = data[: , 1:]
    len_obs = len(data[0])

    Exo_datasets = ["Exoplanets_SleepEEG", "Exoplanets_FD-A", "Exoplanets_HAR", "Exoplanets_ECG"]
    TSlength_aligneds = [178, 5120, 206, 1500]  # needed for zero padding

    print('_'*50)
    print(training_mode)

    for n_pre in range(len(Exo_datasets)):
        
        TSlength = TSlength_aligneds[n_pre]
        n_obs = len(data)

        if training_mode == 'test':
            if TSlength < len_obs: # subsampling
            
                num_subsamples = len_obs//TSlength + 1 # last sample completed w padding
                n_pad = ((num_subsamples)*TSlength)-len_obs
                pad_matrix = np.zeros((n_obs,n_pad))
                data_pad = np.hstack((data,pad_matrix))

                keep_indices = np.arange(num_subsamples-1, data_pad.shape[1], num_subsamples) #### CHECK THIS

                # Select the desired columns from the data
                final_data = data_pad[:, keep_indices]
                final_labels = labels

            else: # zero padding
                pad_matrix = np.zeros((len(data), TSlength-len_obs))
                final_data = np.hstack((data, pad_matrix))
                final_labels = labels

        else:
            if TSlength < len_obs: # subsampling
                
                num_subsamples = len_obs//TSlength + 1 # last sample completed w padding

                # pad to make reshape possible
                n_pad = ((num_subsamples)*TSlength)-len_obs
                pad_matrix = np.zeros((n_obs,n_pad))
                data_pad = np.hstack((data,pad_matrix))

                # plan B:
                # reshape((x,num_subsamples))   x = TSlength * len(data)
                temp = np.reshape(data_pad, (TSlength*n_obs , num_subsamples)) # every obs is broken into multiple rows

                # transpose
                temp = np.transpose(temp) # now, every row consists of multiple TSlength-windows

                # reshape ((x,TSlength))  x = num_subsamples * n_obs
                final_data = np.reshape(temp, (num_subsamples*n_obs , TSlength)) # break into seperate TSlength-windows

                # labels = [i for _ in range(5) for i in labels]
                final_labels = np.array([i for _ in range(num_subsamples) for i in labels]) # (1,2,3) --> (1,2,3, 1,2,3, 1,2,3)

            else: # zero padding
                pad_matrix = np.zeros((len(data), TSlength-len_obs))
                final_data = np.hstack((data, pad_matrix))
                final_labels = labels 

        data_matrix = np.hstack((final_labels.reshape(-1,1), final_data))

        # Shuffle the segmented matrix with labels
        np.random.seed(42)
        np.random.shuffle(data_matrix)

        # Separate the features and labels for train, validation, and test sets
        data_shuffle = data_matrix[:, 1:]
        labels_shuffle = data_matrix[:, 0]

        # Convert features and labels to Torch tensors
        samples_tensor = torch.tensor(data_shuffle).unsqueeze(1)
        labels_tensor = torch.tensor(labels_shuffle)

        # Create dictionaries for train, validation, and test data
        train_data_dict = {"samples": samples_tensor, "labels": labels_tensor}

        print(Exo_datasets[n_pre], training_mode, samples_tensor.shape)
        # Save Torch tensors as PT files in new created folders
        if not os.path.exists(f"datasets\\{Exo_datasets[n_pre]}"):
            os.mkdir(f"datasets\\{Exo_datasets[n_pre]}")
        torch.save(train_data_dict, f'datasets\\{Exo_datasets[n_pre]}\\{training_mode}.pt')
    