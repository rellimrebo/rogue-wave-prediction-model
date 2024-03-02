#!pip install netCDF4 pandas
#!pip install pyarrow
#!pip install fastparquet
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import netCDF4
import numpy as np
from datetime import datetime, timedelta
import matplotlib.dates as mpldts
import calendar
from scipy.signal import find_peaks
import random
from random import randint
import time
import pandas as pd
import pyarrow.parquet as pq
import os

qc_level = 2

def get_unix_timestamp(datetime_obj):
    return int(datetime_obj.timestamp())

def get_displacement_data(station, deployment, start_date, end_date):
    # Format the deployment number with a leading zero
    deployment_str = f"{int(deployment):02d}"  # Converts '1' to '01', '2' to '02', etc.
    
    # Construct the file path for the netCDF file based on station and formatted deployment
    data_url = f'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/{station}p1/{station}p1_d{deployment_str}.nc'
    
    # Open the netCDF file using netCDF4.Dataset
    with netCDF4.Dataset(data_url) as nc:
        nc.set_auto_mask(False)
        time_var = nc.variables['waveTime'][:]  # Assuming this is in UNIX timestamp format
        qc_flag = nc.variables['xyzFlagPrimary'][:]
        
        # Convert start and end dates to UNIX timestamps
        unix_start = get_unix_timestamp(start_date)
        unix_end = get_unix_timestamp(end_date)

        # Find the indices for the start and end dates
        start_index = np.searchsorted(time_var, unix_start)
        end_index = np.searchsorted(time_var, unix_end, side='right')
        
        # Filter based on QC flag and start/end indices
        displacement_data = nc.variables['xyzZDisplacement'][start_index:end_index]
        qc_data = qc_flag[start_index:end_index]
        
        # Filter out data that does not meet the QC level
        displacement_data = np.ma.masked_where(qc_data > 2, displacement_data).compressed()
        
        print(f"Station {station}_{deployment}: Data obtained @", datetime.now().strftime("%H:%M:%S"))
        return displacement_data


def find_troughs_and_crests(displacement_data):
    crests, _ = find_peaks(displacement_data)
    troughs, _ = find_peaks(-displacement_data)
    return troughs, crests

def find_nearest_trough(troughs, crest):
    preceding_troughs = troughs[troughs < crest]
    return preceding_troughs.max() if preceding_troughs.size > 0 else None

def calculate_wave_heights(displacement_data, troughs, crests):
    wave_heights = []
    for crest in crests:
        nearest_trough = find_nearest_trough(troughs, crest)
        if nearest_trough is not None:
            height = displacement_data[crest] - displacement_data[nearest_trough]
            wave_heights.append((nearest_trough, crest, height))
    return wave_heights

def calculate_zero_upcrossings(displacement_data, sample_rate):
    zero_crossings = np.where(np.diff(np.sign(displacement_data)) > 0)[0]
    return len(zero_crossings), zero_crossings, np.mean(np.diff(zero_crossings)) / sample_rate if len(zero_crossings) > 1 else 0

def repetitive_value_checking(displacement_data):
    for i in range(len(displacement_data) - 10 + 1):
        if all(np.abs(displacement_data[i:i+10]) > 20.47):
            return True  # Indicates repetitive values exceeding threshold found
    return None

def excess_sensor_limit(displacement_data):
    if np.any(np.abs(displacement_data) > 20.47):
        return None

def should_discard_block(displacement_data, sample_rate, threshold):
    rate_of_change = np.abs(np.diff(displacement_data)) / (1/sample_rate)
    print("rate",rate_of_change)
    print("thresh",threshold)
    return np.any(rate_of_change > threshold)

def detect_non_rogue_wave(displacement_data, sample_rate, sigma):

    N_z, zero_upcrossing_indices, T_z = calculate_zero_upcrossings(displacement_data, sample_rate)
    if T_z == 0:  # Prevent division by zero
        return None
    S_y = (4 * sigma / T_z) * np.sqrt(2 * np.log(N_z))

    #print("checking if need to discard block")
    # Discard the block if the threshold is exceeded
    #print(f"{should_discard_block(displacement_data, sample_rate, S_y)} {repetitive_value_checking(displacement_data)} {excess_sensor_limit(displacement_data)}")
    if should_discard_block(displacement_data, sample_rate, S_y) or repetitive_value_checking(displacement_data) or excess_sensor_limit(displacement_data):
        # print("Measurement discarded due to exceeding the rate of change threshold or repetitive)
        
        return None

    # Calculate the significant wave height (Hs)
    Hs = 4 * sigma

    # Calculate the troughs and crests in the window
    #print("calculate measurements")
    troughs, crests = find_troughs_and_crests(displacement_data)
    wave_heights_info = calculate_wave_heights(displacement_data, troughs, crests)

    # Verify all wave heights in the window are less than 2 * Hs
    #print("finished finding rogue waves!")
    for trough, crest, height in wave_heights_info:
        #print("checking height")
        if height >= 2 * Hs:
            #print("found rogue")
            return False  # Window contains wave height >= 2 * Hs, not non-rogue

    # If we get here, all wave heights in the window are less than 2 * Hs
    return True



non_rogue_wave_data = pd.DataFrame(columns=['Station', 'Deployment', 'SamplingRate', 'Segment'])
last_station = None
start_time = time.time()

def process_deployment(station, deployment, start_date, end_date, total_target_blocks, sample_rate):
    non_rogue_blocks = []
    displacement_data = get_displacement_data(station, deployment, start_date, end_date)
    sigma = np.std(displacement_data)

    # Calculate the number of samples needed for a 30-minute segment
    num_samples_for_30_min = int(30 * 60 * sample_rate)

    # Verify there's enough data for at least one 30-minute segment
    if len(displacement_data) < num_samples_for_30_min:
        print(f"Not enough data for a 30-minute segment. Only {len(displacement_data)} samples are available.")
        print(f"Breaking out of: {station}_{deployment}")
        return non_rogue_blocks  # Return empty list or handle accordingly

    while_stop = 0
    while len(non_rogue_blocks) < total_target_blocks:
        start_index = random.randint(0, len(displacement_data) - num_samples_for_30_min)
        end_index = start_index + num_samples_for_30_min
        segment = displacement_data[start_index:end_index]
        if detect_non_rogue_wave(segment, sample_rate, sigma):
            non_rogue_blocks.append(segment.tolist())  # Convert to list for memory efficiency
            if len(non_rogue_blocks) == total_target_blocks:
                break
        else:
            while_stop += 1
            #print(detect_non_rogue_wave(segment, sample_rate, sigma))
        if while_stop >= 100000:
            print(f"while loop Hard Deck REACHED, Breaking out of: {station}_{deployment}")
            break

    return non_rogue_blocks

def main():
    deployment_info = pd.read_csv('station_deployment_info.csv')  # Ensure correct file path
    bouy_distribution = pd.read_csv('bouy_distribution.csv') #Ensure correct file path
    #file_path = r"data\non_rogue_HB\rogue_wave_data_station_214_new"
    non_rogue_wave_data_list = []
    target_station = 214
    target_station_index = 2 #initialize based on row of bouy_distribution.csv
    total_target_blocks = 0 #initialize based on bouy_distribution.csv
    sample_rate = 1.28  # Sample rate in Hz
    skip_station = False

    #print(bouy_distribution.iloc[2,0])
    #print(bouy_distribution[0])

    ### AUTO MODE
    for _, row in deployment_info.iterrows():
        station = row['Station']
        deployment = row['Deployment']
        target_station = bouy_distribution.loc[target_station_index][0]
        total_target_blocks = bouy_distribution.loc[target_station_index][1]
        file_path = f'data/non_rogue_HB/non_rogue_wave_data_station_{station}.parquet'

        if skip_station and deployment == 1:
            skip_station = False
        if station == target_station and not skip_station:  # Target specific station
            print(station)
            start_date = datetime.strptime(row['Start date'], '%m-%d-%Y %H:%M')
            end_date = datetime.strptime(row['End date'], '%m-%d-%Y %H:%M')
            non_rogue_blocks = process_deployment(station, deployment, start_date, end_date, total_target_blocks, sample_rate)

            # Check if the parquet file already exists
            if os.path.exists(file_path):
                # Read the existing data
                existing_data = pd.read_parquet(file_path)
                # Append new data
                new_data = pd.DataFrame({'NonRogueWaveSegments': non_rogue_blocks})
                appended_data = pd.concat([existing_data, new_data], ignore_index=True)
                # Save the appended data
                appended_data.to_parquet(file_path, index=False)
            else:
                # If the file does not exist, just save the new data
                new_data = pd.DataFrame({'NonRogueWaveSegments': non_rogue_blocks})
                new_data.to_parquet(file_path, index=False)


            # Check if we have reached the target number of blocks
            if len(non_rogue_blocks) >= total_target_blocks:
                skip_station = True
                target_station_index += 1
                print("Finished Station",station,"@",datetime.now())
                print("End by Break")

    print('Processing completed.')
    print(datetime.now())
    

    ### MANUAL MODE
    '''
    target_station = 143
    total_target_blocks = 573
    file_path = f"data/non_rogue_HB/non_rogue_wave_data_station_{target_station}.parquet"

    for _, row in deployment_info.iterrows():
        station = row['Station']
        deployment = row['Deployment']

        if station == target_station:  # Target specific station
            start_date = datetime.strptime(row['Start date'], '%m-%d-%Y %H:%M')
            end_date = datetime.strptime(row['End date'], '%m-%d-%Y %H:%M')
            non_rogue_blocks = process_deployment(station, deployment, start_date, end_date, total_target_blocks, sample_rate)

            # Check if the parquet file already exists
            if os.path.exists(file_path):
                # Read the existing data
                existing_data = pd.read_parquet(file_path)
                # Append new data
                new_data = pd.DataFrame({'NonRogueWaveSegments': non_rogue_blocks})
                appended_data = pd.concat([existing_data, new_data], ignore_index=True)
                # Save the appended data
                appended_data.to_parquet(file_path, index=False)
            else:
                # If the file does not exist, just save the new data
                new_data = pd.DataFrame({'NonRogueWaveSegments': non_rogue_blocks})
                new_data.to_parquet(file_path, index=False)

            # Check if we have reached the target number of blocks
            if len(non_rogue_blocks) >= total_target_blocks:
                break

    print('Processing completed.')
    '''
        

if __name__ == "__main__":
    main()
