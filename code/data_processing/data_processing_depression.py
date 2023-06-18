import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from process_functions import check_consecutive_repeats

## Processing the Depression dataset and saving training and test set as pt files ##

np.random.seed(12)
path = os.getcwd()
cond = f'{path}\datasets\Depression\data\condition'
cont = f'{path}\datasets\Depression\data\control'
combined_data = pd.DataFrame()

# create specific dataset for each pre_trained model
Depression_datasets = ["Depression_SleepEEG", "Depression_FD_A", "Depression_HAR", "Depression_ECG"]
TSlength_aligneds = [178, 5120, 206, 1500]  # needed for zero padding
Segments_days_min = [(1, 8), (3, 1), (1, 7), (1, 1)]  # SLEEP = 180, FD_A = 4320, HAR = 205,7,  ECG = 1440

for j in range(len(Depression_datasets)):
    num_days, num_min = Segments_days_min[j]

    segment_length = 60*24 * num_days
    segmented_data_train = []
    segmented_data_test = []
    labels_train = []
    labels_test = []

    # Load and append data from the control and condition folder
    folders = [cont, cond]
    for folder, label in zip(folders, [0, 1]):
        files = os.listdir(folder)
        np.random.shuffle(files)
        train_files, test_files = train_test_split(files, test_size=0.75, random_state=12)

        file_types = [train_files, test_files]

        for file_type in file_types:

            for file in file_type:
                file_path = os.path.join(folder, file)
                DATA = pd.read_csv(file_path)
                start_index = 0

                ## Finding the first 00:00:00 timestamp
                for i in range(len(DATA["timestamp"])):
                    if DATA["timestamp"][i][11:19] == '00:00:00':
                        start_index = i
                        break
                data = DATA["activity"][start_index:]


                # Segment the data into days and remove days with consecutive repeats and low activity
                data_seg_days = []
                num_segments = len(data) // 1440  # Number of segments for this file
                for i in range(num_segments):
                    start_index = i * 1440
                    end_index = start_index + 1440
                    day_segments = data.iloc[start_index:end_index]
                    if day_segments.mean() > 10 and check_consecutive_repeats(day_segments, 200) == False:
                        data_seg_days.append(day_segments)

                clean_df = pd.concat(data_seg_days, axis=0, ignore_index=True)


                # Segment the signal and add to the segmented_data list
                num_segments = len(clean_df) // segment_length  # Number of segments in the clean dataframe
                for i in range(num_segments):
                    start_index = i * segment_length
                    end_index = start_index + segment_length
                    segment = clean_df.iloc[start_index:end_index].values

                    # SQUEEZING SEGMENTS
                    new_segment = []

                    for i in range(len(segment)//num_min):
                        start_index = i * num_min
                        end_index = start_index + num_min
                        new_segment.append(sum(segment[start_index:end_index]))

                    # ZERO PADDING
                    if len(new_segment) < TSlength_aligneds[j]:
                        zero_pads = TSlength_aligneds[j] - len(new_segment)
                        for _ in range(zero_pads): new_segment.append(0)

                    if file_type == file_types[0]:  # filetypes[0] = train
                        segmented_data_train.append(new_segment)
                        labels_train.append(label)
                    else:
                        segmented_data_test.append(new_segment)
                        labels_test.append(label)


    # Data matrix with labels
    segmented_matrix_train = np.vstack(segmented_data_train)
    segmented_matrix_test = np.vstack(segmented_data_test)
    label_train_column = np.array(labels_train).reshape(-1, 1)
    label_test_column = np.array(labels_test).reshape(-1, 1)

    data_matrix_train = np.hstack((label_train_column, segmented_matrix_train))
    data_matrix_test = np.hstack((label_test_column, segmented_matrix_test))

    # Shuffle the segmented matrix with labels
    np.random.shuffle(data_matrix_train)
    np.random.shuffle(data_matrix_test)

    # Separate the features and labels for train and test sets
    train_features = data_matrix_train[:, 1:]
    train_labels = data_matrix_train[:, 0]
    test_features = data_matrix_test[:, 1:]
    test_labels = data_matrix_test[:, 0]

    # Convert features and labels to Torch tensors
    train_samples_tensor = torch.tensor(train_features).unsqueeze(1)
    train_labels_tensor = torch.tensor(train_labels)
    test_samples_tensor = torch.tensor(test_features).unsqueeze(1)
    test_labels_tensor = torch.tensor(test_labels)

    # Create dictionaries for train and test data
    train_data_dict = {"samples": train_samples_tensor, "labels": train_labels_tensor}
    test_data_dict = {"samples": test_samples_tensor, "labels": test_labels_tensor}

    print(Depression_datasets[j],train_samples_tensor.shape, test_samples_tensor.shape)
    # Save Torch tensors as PT files in new created folders
    if not os.path.exists(f"datasets\\{Depression_datasets[j]}"):
        os.mkdir(f"datasets\\{Depression_datasets[j]}")
    torch.save(train_data_dict, f'datasets\\{Depression_datasets[j]}\\train.pt')
    torch.save(test_data_dict, f'datasets\\{Depression_datasets[j]}\\test.pt')