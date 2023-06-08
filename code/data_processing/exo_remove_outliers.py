import numpy as np

def remove_outliers(matrix):
    processed_matrix = np.copy(matrix)  # Create a copy of the input matrix
    
    for i in range(matrix.shape[0]):
        row = matrix[i, :]
        row_maxidx, row_minidx = row.argmax(), row.argmin()
        std = np.std(row)
        mean = np.mean(row)

        
        # Update values exceeding the threshold
        processed_matrix[i, row_maxidx] = mean
        processed_matrix[i, row_minidx] = mean
    
    return processed_matrix