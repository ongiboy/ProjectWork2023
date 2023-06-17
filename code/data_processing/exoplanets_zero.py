import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from process_functions import *

path = os.getcwd()
path_train = f'{path}\datasets\Exoplanets\exoTrain.csv'
path_test = f'{path}\datasets\Exoplanets\exoTest.csv'

df_train = pd.read_csv(path_train)
df_test = pd.read_csv(path_test)
df = df_train.append(df_test, ignore_index=True) # combine into one data set

data = df.values
labels = data[:,0]-1
data = data[: , 1:]
len_obs = len(data[0])
TSlength = 5120

data, labels = remove_outliers(data, labels)


pad_matrix = np.zeros((len(data), TSlength-len_obs))
padded_data = np.hstack((data, pad_matrix))

padded_df = pd.DataFrame(padded_data)
padded_df.insert(0, "LABEL", labels)

# Separate the DataFrame into two classes
nonexo_df = padded_df[padded_df['LABEL'] == 0]
exo_df = padded_df[padded_df['LABEL'] == 1]

# Training set is 60 non-exo planets and 30 exo planets
train_nonexo = nonexo_df.sample(n=40, random_state=1)
nonexo_df = nonexo_df.drop(train_nonexo.index)

train_exo = exo_df.sample(n=20, random_state=1)
exo_df = exo_df.drop(train_exo.index)

# Test set is the remaining rows
test_nonexo = nonexo_df
test_exo = exo_df

# Concatenate the rows from both classes for each dataset
train_combined = pd.concat([train_nonexo, train_exo])
test_combined = pd.concat([test_nonexo, test_exo])

# Shuffle the rows of each dataset
np.random.seed(42)
train_shuffled = train_combined.sample(frac=1, replace=False)
test_shuffled = test_combined.sample(frac=1, replace=False)


train_values = train_combined.values[: , 1:]
train_labels = train_combined.values[: , 0]
test_values = test_combined.values[: , 1:]
test_labels = test_combined.values[: , 0]


# Convert features and labels to Torch tensors
train_samples_tensor = torch.tensor(train_values).unsqueeze(1)
train_labels_tensor = torch.tensor(train_labels)
test_samples_tensor = torch.tensor(test_values).unsqueeze(1)
test_labels_tensor = torch.tensor(test_labels)

# Create dictionaries for train, validation, and test data
train_data_dict = {"samples": train_samples_tensor, "labels": train_labels_tensor}
test_data_dict = {"samples": test_samples_tensor, "labels": test_labels_tensor}

print('Training size: ', train_samples_tensor.shape, train_labels_tensor.shape)
print('Test size: ', test_samples_tensor.shape, test_labels_tensor.shape)
# Save Torch tensors as PT files in new created folders
if not os.path.exists(f"datasets\\Exoplanets_FD_A"):
    os.mkdir(f"datasets\\Exoplanets_FD_A")
torch.save(train_data_dict, f'datasets\\Exoplanets_FD_A\\train.pt')
torch.save(test_data_dict, f'datasets\\Exoplanets_FD_A\\test.pt')




