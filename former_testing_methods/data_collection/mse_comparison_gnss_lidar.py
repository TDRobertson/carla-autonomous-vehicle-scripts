# Description: This script calculates the mean squared error (MSE) between the control and test data for GNSS and LiDAR data.
# The script reads the control and test data from CSV files and calculates the MSE for each column of interest over a specified time period.
# The script writes the results to a CSV file with the format: Data Type,Test File,Column,MSE.

import pandas as pd
import os

# Define file paths for GNSS data
# Control file to compare against
gnss_control_file = 'gnss_data.csv'
# Test files to compare against control file
gnss_test_files = []

# Define file paths for LiDAR data
# Control file to compare against
lidar_control_file = 'lidar_data.csv'
# Test files to compare against control file
lidar_test_files = []

# Load control files
gnss_control_df = pd.read_csv(gnss_control_file)
lidar_control_df = pd.read_csv(lidar_control_file)

# Specify columns of interest
gnss_columns_of_interest = ['latitude', 'longitude', 'altitude']
lidar_columns_of_interest = ['x', 'y', 'z', 'intensity']
timestamp_column = 'timestamp_sec'

# Function to calculate mean squared error over a specified time period for a specific column
def calculate_mse_column(df1, df2, column_name, time_period):
    df1 = df1[(df1[timestamp_column] >= time_period[0]) & (df1[timestamp_column] <= time_period[1])]
    df2 = df2[(df2[timestamp_column] >= time_period[0]) & (df2[timestamp_column] <= time_period[1])]

    if column_name not in df1.columns or column_name not in df2.columns:
        return float('nan')
    
    merged_df = pd.merge(df1[[timestamp_column, column_name]], df2[[timestamp_column, column_name]], on=timestamp_column, suffixes=('_control', '_test'))
    mse = ((merged_df[f'{column_name}_control'] - merged_df[f'{column_name}_test']) ** 2).mean()
    
    return mse

# Specify the time period (start_time, end_time)
time_period = (10, 20)  # Adjust the time period as needed

# Open the output file
with open('mse_column_results.csv', 'w') as output_file:
    output_file.write('Data Type,Test File,Column,MSE\n')
    
    # Iterate over each GNSS test file
    for test_file in gnss_test_files:
        test_df = pd.read_csv(test_file)
        
        # Calculate mean squared error for each column
        for column_name in gnss_columns_of_interest:
            mse = calculate_mse_column(gnss_control_df, test_df, column_name, time_period)
            output_file.write(f"GNSS,{test_file},{column_name},{mse}\n")
    
    # Iterate over each LiDAR test file
    for test_file in lidar_test_files:
        test_df = pd.read_csv(test_file)
        
        # Calculate mean squared error for each column
        for column_name in lidar_columns_of_interest:
            mse = calculate_mse_column(lidar_control_df, test_df, column_name, time_period)
            output_file.write(f"LiDAR,{test_file},{column_name},{mse}\n")
