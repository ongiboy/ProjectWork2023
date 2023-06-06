# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 13:26:21 2023

@author: kavus
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

path = os.getcwd()
cond = f'{path}\datasets\Depression\data\condition'
cont = f'{path}\datasets\Depression\data\control'
combined_data = pd.DataFrame()

# defining data
num_days = 1
num_min = 8
segment_length = 60*24 * num_days
segmented_data = []
labels = []

# Load and append data from the control folder
folders = [cont, cond]
for folder, label in zip(folders, [0, 1]):
    files = os.listdir(folder)
    for file in files:
        file_path = os.path.join(folder, file)
        DATA = pd.read_csv(file_path)
        start_index = 0
        for i in range(len(DATA["timestamp"])):
            if DATA["timestamp"][i][11:19] == '00:00:00':
                start_index = i
                break
        data = DATA["activity"][start_index:]


        num_segments = len(data) // segment_length  # Number of segments for this file

        # Segment the signal and add to the segmented_data list
        for i in range(num_segments):
            start_index = i * segment_length
            end_index = start_index + segment_length
            segment = data.iloc[start_index:end_index].values

            new_segment = []

            for i in range(len(segment)//num_min):
                start_index = i * num_min
                end_index = start_index + num_min
                new_segment.append(sum(segment[start_index:end_index]))

            segmented_data.append(new_segment)
            labels.append(label)


# Data matrix with labels
segmented_matrix = np.vstack(segmented_data)
label_column = np.array(labels).reshape(-1, 1)
data_matrix = np.hstack((label_column, segmented_matrix))

# Shuffle the segmented matrix with labels
np.random.seed(42)
np.random.shuffle(data_matrix)

# Split the data into train and remaining data
train_data, remaining_data = train_test_split(data_matrix, test_size=0.75, stratify=data_matrix[:, 0], random_state=42)

# Split the remaining data into validation and test
val_data, test_data = train_test_split(remaining_data, test_size=0.89, stratify=remaining_data[:, 0], random_state=42)

# Separate the features and labels for train, validation, and test sets
train_features = train_data[:, 1:]
train_labels = train_data[:, 0]
val_features = val_data[:, 1:]
val_labels = val_data[:, 0]
test_features = test_data[:, 1:]
test_labels = test_data[:, 0]

# Convert features and labels to Torch tensors
train_samples_tensor = torch.tensor(train_features).unsqueeze(1)
train_labels_tensor = torch.tensor(train_labels)
val_samples_tensor = torch.tensor(val_features).unsqueeze(1)
val_labels_tensor = torch.tensor(val_labels)
test_samples_tensor = torch.tensor(test_features).unsqueeze(1)
test_labels_tensor = torch.tensor(test_labels)

# Create dictionaries for train, validation, and test data
train_data_dict = {"samples": train_samples_tensor, "labels": train_labels_tensor}
val_data_dict = {"samples": val_samples_tensor, "labels": val_labels_tensor}
test_data_dict = {"samples": test_samples_tensor, "labels": test_labels_tensor}


# Save Torch tensors to PT files
# Save the dictionaries as PT files
torch.save(train_data_dict, 'datasets\\Depression\\train.pt')
torch.save(val_data_dict, 'datasets\\Depression\\val.pt')
torch.save(test_data_dict, 'datasets\\Depression\\test.pt')