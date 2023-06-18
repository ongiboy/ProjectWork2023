from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
import torch
import numpy as np

## Trains and tests a KNN classifier on the datasets using cross validation##

datasets = ["SleepEEG", #0
            "Epilepsy", 

            "FD_A", #2
            "FD_B",

            "ECG", #4
            "EMG",

            "HAR", #6
            "Gesture",
            
            "Depression_SleepEEG", #8
            "Depression_FD_A",
            "Depression_HAR", #10
            "Depression_ECG"]

dataset = datasets[8]


# Import data
train = torch.load(f'datasets\\{dataset}\\train.pt')
x_train, y_train = np.squeeze(train["samples"]), train["labels"]
test = torch.load(f'datasets\\{dataset}\\test.pt')
x_test, y_test = np.squeeze(test["samples"]), test["labels"]

# KNN
n_neighbors = [1,3,5,7,9]
accuracies = []
for n in n_neighbors: # find optimal n
    knn = KNeighborsClassifier(n_neighbors=n)
    
    y_pred = cross_val_predict(knn, np.squeeze(x_train), y_train, cv=5)  # You can adjust the number of folds as needed
    balanced_acc = balanced_accuracy_score(y_train, y_pred)
    accuracies.append(balanced_acc)

# use optimal n
n_opt = n_neighbors[np.argmax(accuracies)]
knn_final = KNeighborsClassifier(n_neighbors=n_opt)
knn_final.fit(x_train, y_train)

# Evaluate accuracy
y_pred = knn_final.predict(x_test)
accuracy = balanced_accuracy_score(y_test, y_pred)

print(accuracy)