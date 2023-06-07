import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

path = os.getcwd()
path_train = f'{path}\datasets\Exoplanets\exoTrain.csv'
path_test = f'{path}\datasets\Exoplanets\exoTest.csv'


df_train = pd.read_csv(path_train)
df_test = pd.read_csv(path_test)
df = df_train.append(df_test, ignore_index=True) # combine into one data set
data = np.random.permutation(df.values) # scramble data (rows)

# extract labels
labels = data[:,0]-1 
data = data[: , 1:]
len_obs = len(data[0])

Exo_datasets = ["Exoplanets_SleepEEG", "Exoplanets_FD-A", "Exoplanets_HAR", "Exoplanets_ECG"]
TSlength_aligneds = [178, 5120, 206, 1500]  # needed for zero padding
final_data = np.array([])
for n_pre in range(len(Exo_datasets)):

    TSlength = TSlength_aligneds[n_pre]
    n_obs = len(data)
    
    if TSlength < len_obs: # subsampling
        
        num_subsamples = len_obs//TSlength + 1 # last sample completed w padding

        # pad to make reshape possible
        n_pad = ((num_subsamples)*TSlength)-len_obs
        pad_matrix = np.zeros((n_obs,n_pad))
        data = np.hstack((data,pad_matrix))

        # plan B:
        # reshape((x,num_subsamples))   x = TSlength * len(data)
        temp = np.reshape(data, (TSlength*n_obs , num_subsamples)) # every obs is broken into multiple rows

        # transpose
        temp = np.transpose(temp) # now, every row consists of multiple TSlength-windows

        # reshape ((x,TSlength))  x = num_subsamples * n_obs
        final_data = np.reshape(temp, (num_subsamples*n_obs , TSlength)) # break into seperate TSlength-windows

        # labels = [i for _ in range(5) for i in labels]
        labels = np.array([i for _ in range(num_subsamples) for i in labels]) # (1,2,3) --> (1,2,3, 1,2,3, 1,2,3)
        """ for i in range(30):
            fig,axs = plt.subplots(2)
            axs[0].plot(final_data[i])
            axs[1].plot(data[i])
            plt.show()
            print()
            plt.close()
        print() """

        """ for i in range(len(data)):
            obs = data[i]
            print("debug: ", i, obs.shape)
            temp = np.reshape(obs, (TSlength , num_subsamples))
            temp = np.transpose(temp)
            if i == 0:
                final_data = temp
            else:
                final_data = np.vstack((final_data,temp))

        final_labels = np.array([i for i in labels for _ in range(5)]) """

    else: # zero padding
        pad_matrix = np.zeros((len(data), TSlength-len_obs))
        final_data = np.hstack((data, pad_matrix))
        final_labels = labels
        print()    

final = torch.tensor(final_data).unsqueeze(1)

