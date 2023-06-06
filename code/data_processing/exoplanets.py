import pandas as pd
import numpy as np
import torch
import os

path = os.getcwd()
path_train = f'{path}\datasets\Exoplanets\exoTrain.csv'
path_test = f'{path}\datasets\Exoplantes\exoTest.csv'


df_train = pd.read_csv(path_train)
df_test = pd.read_csv(path_test)


labels_train = df_train['LABEL']
labels_test = df_test['LABEL']

values = df_train.iloc[:, 2:].values

# Convert the values to a NumPy matrix
matrix = np.matrix(values)

# Print the matrix shape
train_samples_tensor = torch.tensor(matrix).unsqueeze(1)

print(df_train)

