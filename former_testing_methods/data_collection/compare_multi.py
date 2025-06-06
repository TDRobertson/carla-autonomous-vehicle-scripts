# The purpose of this script is to compare the confidence loss of the specified classes between a control file and multiple test files gathered from Carla Simulator.
import pandas as pd
import os

# Define file paths
control_file = 'detections_0C.csv'
test_files = [
    'detections_50C.csv', 
    'detections_100C.csv', 
    'detections_150C.csv', 
    'detections_200C.csv', 
    'detections_300C.csv'
]

# Load control file
control_df = pd.read_csv(control_file)

# Filter only the specified classes
classes_of_interest = ['stop sign', 'bench', 'traffic light', 'fire hydrant', 'car']
control_df = control_df[control_df['class'].isin(classes_of_interest)]

# Function to calculate confidence loss over a specified time period
def calculate_confidence_loss(df1, df2, time_period, class_name):
    df1_class = df1[df1['class'] == class_name]
    df2_class = df2[df2['class'] == class_name]

    df1_class = df1_class[(df1_class['timestamp'] >= time_period[0]) & (df1_class['timestamp'] <= time_period[1])]
    df2_class = df2_class[(df2_class['timestamp'] >= time_period[0]) & (df2_class['timestamp'] <= time_period[1])]

    if len(df1_class) == 0 or len(df2_class) == 0:
        return float('nan')

    confidence_loss = df1_class['confidence'].mean() - df2_class['confidence'].mean()
    return confidence_loss

# Specify the time period to begin comparison of timestamped data (start_time, end_time)
time_period = (300, 800) 

# Open the output file
with open('confidence_loss_results.csv', 'w') as output_file:
    output_file.write('Test File,Class,Confidence Loss\n')
    
    # Iterate over each test file
    for test_file in test_files:
        test_df = pd.read_csv(test_file)
        test_df = test_df[test_df['class'].isin(classes_of_interest)]
        
        # Calculate confidence loss for each class
        for class_name in classes_of_interest:
            loss = calculate_confidence_loss(control_df, test_df, time_period, class_name)
            output_file.write(f"{test_file},{class_name},{loss}\n")
