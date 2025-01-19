#Generated functions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_datetime_with_nan(df, column_name='DateTime'):
    """
    Plots the differences in Unix Epoch Time for the 'DateTime' column in the given DataFrame.
    NaN values are marked in red with larger markers.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the 'DateTime' column.
    column_name (str): The name of the column to be processed. Default is 'DateTime'.

    Returns:
    None
    """
    # Convert 'DateTime' column to datetime type
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    
    # Convert to Unix Epoch Time
    epoch_times = df[column_name].apply(lambda x: x.timestamp() if pd.notna(x) else np.nan).values
    
    # Calculate the difference from the previous non-NaN sample
    diff_epoch_times = np.zeros_like(epoch_times)
    for i in range(len(epoch_times)):
        if not np.isnan(epoch_times[i]):
            # Find the previous non-NaN sample
            j = i - 1
            while j >= 0 and np.isnan(epoch_times[j]):
                j -= 1
            if j >= 0:
                diff_epoch_times[i] = epoch_times[i] - epoch_times[j]
            else:
                diff_epoch_times[i] = 0  # No previous valid sample
        else:
            diff_epoch_times[i] = np.nan
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    
    # Mark NaN values in red with larger markers
    nan_indices = np.where(np.isnan(diff_epoch_times))[0]
    nan_y_values = np.zeros_like(nan_indices)  # Assign a placeholder value for NaN y-coordinates
    plt.scatter(nan_indices, nan_y_values, color='red', label='NaN values', s=100)  # s=100 for larger markers
    
    # Plot non-NaN values with alpha=0.2
    non_nan_indices = np.where(~np.isnan(diff_epoch_times))[0]
    plt.plot(non_nan_indices, diff_epoch_times[non_nan_indices], label='DateTime Differences', alpha=0.2)
    
    plt.xlabel('Sample Index')
    plt.ylabel('Difference in Unix Epoch Time')
    plt.title('DateTime Column Differences with NaN Values Marked in Red')
    plt.xlim(350,1000)
    plt.show()

def create_nan_context_matrix(df, column_name='DateTime', M=5, indices_to_plot=None, plot_differences=False):
    """
    Creates a context matrix for NaN values in the 'DateTime' column, with M values before and M values after each NaN.
    Optionally plots the specified indices.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the 'DateTime' column.
    column_name (str): The name of the column to be processed. Default is 'DateTime'.
    M (int): The number of values to take before and after each NaN. Default is 5.
    indices_to_plot (list): List of indices to plot. Default is None.
    plot_differences (bool): Whether to plot differences in Unix Epoch Time instead of the raw values. Default is False.

    Returns:
    np.ndarray: The context matrix with shape (N, 2M+1), where N is the number of NaNs.
    """
    # Convert 'DateTime' column to datetime type
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    
    # Convert to Unix Epoch Time
    epoch_times = df[column_name].apply(lambda x: x.timestamp() if pd.notna(x) else np.nan).values
    
    # Find indices of NaN values
    nan_indices = np.where(np.isnan(epoch_times))[0]
    
    # Initialize the matrix
    context_matrix = np.full((len(nan_indices), 2 * M + 1), np.nan)
    
    # Fill the matrix
    for i, idx in enumerate(nan_indices):
        start_idx = max(0, idx - M)
        end_idx = min(len(epoch_times), idx + M + 1)
        context = epoch_times[start_idx:end_idx]
        
        # Ensure the context has exactly 2M+1 elements
        if len(context) < 2 * M + 1:
            context = np.pad(context, (M - idx if idx < M else 0, 2 * M + 1 - len(context)), constant_values=np.nan)
        
        context_matrix[i] = context
    
    # Calculate differences if required
    if plot_differences:
        diff_context_matrix = np.zeros_like(context_matrix)
        for i in range(context_matrix.shape[0]):
            for j in range(1, context_matrix.shape[1]):
                if not np.isnan(context_matrix[i, j]) and not np.isnan(context_matrix[i, j - 1]):
                    diff_context_matrix[i, j] = context_matrix[i, j] - context_matrix[i, j - 1]
                else:
                    diff_context_matrix[i, j] = np.nan
        context_matrix = diff_context_matrix
    
    # Plot the specified indices
    if indices_to_plot is not None:
        plt.figure(figsize=(10, 6))
        for idx in indices_to_plot:
            if idx < len(context_matrix):
                plt.plot(context_matrix[idx], label=f'NaN Index {nan_indices[idx]}')
        plt.xlabel('Relative Index')
        plt.ylabel('Difference in Unix Epoch Time' if plot_differences else 'Unix Epoch Time')
        plt.title('Context of NaN Values')
        plt.legend(loc = "upper right")
        plt.show()
    
    return context_matrix

# Example usage with the current DataFrame df
plot_datetime_with_nan(df.copy())

# Example usage with the current DataFrame df
M = 5
indices_to_plot = np.arange(5,10)  # Example indices to plot
context_matrix = create_nan_context_matrix(df.copy(), M=M, indices_to_plot=indices_to_plot, plot_differences=False)    

