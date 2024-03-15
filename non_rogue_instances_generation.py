import netCDF4
import numpy as np
# import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mpldts
import calendar
from scipy.signal import find_peaks
from random import randint
from scipy.interpolate import interp1d
import time
import pandas as pd
import pyarrow.parquet as pq

# Helper functions
# def on_key(event):
#     if event.key == 'q':
#         plt.close()  # Close the figure window

def find_troughs_and_crests(displacement_data):
    '''
    Finds the wave troughs and crests in a given set of data
    '''
    crests,_ = find_peaks(displacement_data)
    troughs,_ = find_peaks(-displacement_data)

    return troughs, crests

def find_nearest_trough(troughs, crest):
    # Filter troughs to find those before the current crest
    preceding_troughs = troughs[troughs < crest]
    if preceding_troughs.size > 0:
        # Return the nearest trough by picking the maximum of the filtered troughs
        return preceding_troughs.max()
    else:
        # If there are no troughs before this crest, return None
        return None

def calculate_wave_heights(displacement_data, troughs, crests):
    '''
    Calculates the wave heights from a given set of data from
    the given trough and crest indices
    '''
    wave_heights = []
    for crest in crests:
        # Find the nearest tough before the crest
        nearest_trough = find_nearest_trough(troughs, crest)
        if nearest_trough is not None:
            height = displacement_data[crest] - displacement_data[nearest_trough]
            wave_heights.append((nearest_trough, crest, height))
    return wave_heights

def calculate_zero_upcrossings(displacement_data, sample_rate):
    # Find where the displacement data crosses zero from negative to positive
    zero_crossings = np.where(np.diff(np.sign(displacement_data)) > 0)[0]
    return len(zero_crossings), zero_crossings, np.mean(np.diff(zero_crossings)) / sample_rate if len(zero_crossings) > 1 else 0

def should_discard_block(displacement_data, sample_rate, threshold):
    # Calculate rate of change
    # padded_displacement_data = np.pad(displacement_data, (0, 1), mode='edge')
    rate_of_change = np.abs(np.diff(displacement_data)) / (1/sample_rate)

    # Check if any rate of change exceeds the threshold
    if np.any(rate_of_change > threshold):
        return True
    return False

def detect_rogue_wave(displacement_data, sample_rate, sigma):
    '''
    Detects a single rogue wave event in the given displacement data.
    Rogue wave is defined as a wave where H/Hs > Hs_factor,
    where H is the wave height from trough to crest and Hs is the
    significant wave height
    
    Returns the segment of data normalized for the rogue wave event'''

    # Calculate significant wave height Sy and threshold
    # sigma = np.std(displacement_data)
    # sigma = np.std(displacement_data[np.abs(displacement_data) < 20.47])
    N_z, zero_upcrossing_indices, T_z = calculate_zero_upcrossings(displacement_data, sample_rate)
    if T_z == 0:  # Prevent division by zero
        return '1'
    S_y = (4 * sigma / T_z) * np.sqrt(2 * np.log(N_z))
    # Discard the block if the threshold is exceeded
    if should_discard_block(displacement_data, sample_rate, S_y):
        # print("Measurement discarded due to exceeding the rate of change threshold.")
        return '1'

    troughs,crests = find_troughs_and_crests(displacement_data)
    wave_heights_info = calculate_wave_heights(displacement_data, troughs, crests)

    wave_heights = [info[2] for info in wave_heights_info]
    if not wave_heights:
        return '1' # No waves detected
    
    if np.any(np.abs(displacement_data) > 20.47):
        return '1'
    
    for i in range(len(displacement_data) - 10 + 1):
        if all(np.abs(displacement_data[i:i+10]) > 20.47):
            return '1'  # Found 10 consecutive values above the threshold

    # Hs calculation: 4 times the standard deviation of the sea surface elevation
    # Hs = 4 * np.std(displacement_data)
    Hs = 4*np.std(displacement_data[np.abs(displacement_data) < 20.47])

    # Identify rogue waves
    for trough, crest, height in wave_heights_info:
        if height / Hs > 2:
            # Rogue wave detected
            # Calculate the number of samples for the normalization TODO: HARDCODE THIS
            rogue_wave_index = crest

            return rogue_wave_index
        
    return None # No rogue wave detected

# load data
samples_df = pd.read_csv('new_samples_distribution_modified.csv')
print(samples_df)

# initialize rogue-wave storage
non_rogue_wave_data = pd.DataFrame(columns=['Station', 'Deployment','SamplingRate','Segment'])

last_station = None

start_time = time.time()
# main loop
for index, row in samples_df.iterrows():
    count = 0
    station = int(row['Station'])
    if station > 163:
        continue
    station = row.iloc[0]
    station = f"{int(station)}"
    deployment = row.iloc[1]
    deployment = f"{int(deployment):02}"

    number_of_samples = row['Samples']

    # Check if the station has changed
    if last_station is not None and station != last_station:
        # Append rogue wave data to the Parquet file
        file_name = f'data/Non Rogue Wave Data/non_rogue_wave_data_station_{last_station}.parquet'
        non_rogue_wave_data.to_parquet(file_name)
        # Clear rogue_wave_data DataFrame for the next station
        non_rogue_wave_data = pd.DataFrame(columns=['Station', 'Deployment', 'SamplingRate', 'Segment'])
    last_station = station


    # DEBUG INPUT
    # station = '143'
    # deployment = '04'

    # Archive
    data_url = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/' + station + 'p1/' + station + 'p1_d' + deployment + '.nc'

    nc = netCDF4.Dataset(data_url)
    # Turn off auto masking
    nc.set_auto_mask(False)

    z_displacement = nc.variables['xyzZDisplacement'][:] # Vertical displacement
    sample_rate = nc.variables['xyzSampleRate'][:].item() # Hz
    # Test implementation
    block_duration = 30 # Minutes

    block_samples = int(block_duration*60*sample_rate)

    # Filter the displacement data based on the max sensor height condition
    filtered_displacement = z_displacement[np.abs(z_displacement) < 20.47]

    # Calculate the standard deviation for the filtered data
    sigma = np.std(filtered_displacement)

    deployment_count = 0
    while deployment_count != number_of_samples:
        # Randomly selecting start index for a segment
        start_index = randint(0, len(z_displacement) - block_samples)
        end_index = start_index + block_samples

        current_block = z_displacement[start_index:end_index]

        if detect_rogue_wave(current_block, sample_rate, sigma) is None:
            new_row = {
                'Station': station,
                'Deployment': deployment,
                'SamplingRate': sample_rate,
                'Segment': current_block  # Make sure this is defined in your processing code
            }
            non_rogue_wave_data = non_rogue_wave_data.append(new_row, ignore_index=True)
            deployment_count += 1
            count += 1
            print(count)

    print(f"Finished processing station {station} deployment {deployment}")
    print("--- %s seconds ---" % (time.time() - start_time))

# Append rogue wave data to the Parquet file
file_name = f'data/Non Rogue Wave Data/non_rogue_wave_data_station_{last_station}.parquet'
non_rogue_wave_data.to_parquet(file_name)
# Clear rogue_wave_data DataFrame for the next station
non_rogue_wave_data = pd.DataFrame(columns=['Station', 'Deployment', 'SamplingRate', 'Segment'])
print('done')