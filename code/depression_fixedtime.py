import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt


path_condition = 'C:/Users/danie/OneDrive - Danmarks Tekniske Universitet/Dokumenter/4. semester/Fagprojekt/ProjectWork2023/datasets/Depression/data/condition/*.csv'
path_control = 'C:/Users/danie/OneDrive - Danmarks Tekniske Universitet/Dokumenter/4. semester/Fagprojekt/ProjectWork2023/datasets/Depression/data/control/*.csv'

# Get a list of all CSV files in the folder
condition_list = glob.glob(path_condition)
control_list = glob.glob(path_control)

# Initialize an empty list to store the data frames
df_condition_list = []
df_control_list = []

for file in condition_list:
    df = pd.read_csv(file)
    # Check if there are 1440 elements for the same date
    unique_dates = df['date'].unique()
    for date in unique_dates:
        if len(df[df['date'] == date]) == 1440:
            df_condition_list.append(df[df['date'] == date])


for file in control_list:
    df = pd.read_csv(file)
    # Check if there are 1440 elements for the same date
    unique_dates = df['date'].unique()
    for date in unique_dates:
        if len(df[df['date'] == date]) == 1440:
            df_control_list.append(df[df['date'] == date])

# Concatenate all the data frames into a single data frame
combined_condition_df = pd.concat(df_condition_list, axis=0, ignore_index=True)
combined_control_df = pd.concat(df_control_list, axis=0, ignore_index=True)

# Extract the activity scores as a NumPy matrix
condition_activity_scores = combined_condition_df['activity'].to_numpy()
control_activity_scores = combined_control_df['activity'].to_numpy()

# Reshape the activity scores matrix to have each row represent a day
num_days_condition = len(df_condition_list)
num_days_control = len(df_control_list)
condition_activity_scores_matrix = condition_activity_scores.reshape(num_days_condition, 1440)
control_activity_scores_matrix = control_activity_scores.reshape(num_days_control, 1440)


