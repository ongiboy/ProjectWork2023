import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from exo_split_data import split_data
from exo_remove_outliers import remove_outliers

path = os.getcwd()
path_train = f'{path}\datasets\Exoplanets\exoTrain.csv'
path_test = f'{path}\datasets\Exoplanets\exoTest.csv'


# There are 42 exoplanets in the dataset, and we split it 30/6/6
# In training we combine 30 exoplanets with 60 non-exoplanets

df_train = pd.read_csv(path_train)
df_test = pd.read_csv(path_test)
df = df_train.append(df_test, ignore_index=True) # combine into one data set

# Separate the DataFrame into two classes
nonexo_df = df[df['LABEL'] == 1]
exo_df = df[df['LABEL'] == 2]

# Training set is 60 non-exo planets and 30 exo planets
train_nonexo = nonexo_df.sample(n=60, random_state=1)
nonexo_df = nonexo_df.drop(train_nonexo.index)

train_exo = exo_df.sample(n=30, random_state=1)
exo_df = exo_df.drop(train_exo.index)

# Validation set is 26 non-exo planets and 6 exo planets
valid_nonexo = nonexo_df.sample(n=26, random_state=1)
nonexo_df = nonexo_df.drop(valid_nonexo.index)

valid_exo = exo_df.sample(n=6, random_state=1)
exo_df = exo_df.drop(valid_exo.index)

# Test set is the remaining rows
test_nonexo = nonexo_df
test_exo = exo_df

# Concatenate the rows from both classes for each dataset
train_combined = pd.concat([train_nonexo, train_exo])
valid_combined = pd.concat([valid_nonexo, valid_exo])
test_combined = pd.concat([test_nonexo, test_exo])


# Setting outliers to the mean of the sample
train_values = train_combined.values[: , 1:]
valid_values = valid_combined.values[: , 1:]
test_values = test_combined.values[: , 1:]

train_df_clean = remove_outliers(train_values)
valid_df_clean = remove_outliers(valid_values)
test_df_clean = remove_outliers(test_values)

train_df = pd.DataFrame(train_df_clean)
valid_df = pd.DataFrame(valid_df_clean)
test_df = pd.DataFrame(test_df_clean)


split_data(train_df, "train")
split_data(valid_df, "valid")
split_data(test_df, "test")