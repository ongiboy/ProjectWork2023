import numpy as np

def remove_outliers(matrix):
    processed_matrix = np.copy(matrix)  # Create a copy of the input matrix
    
    for i in range(matrix.shape[0]):
        row = matrix[i, :]
        std = np.std(row)
        mean = np.mean(row)
        threshold = 3 * std
        
        # Update values exceeding the threshold
        processed_matrix[i, row > threshold] = mean
        processed_matrix[i, row < -threshold] = mean
    
    return processed_matrix