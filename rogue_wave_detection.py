'''
TODO

Figure out indexing

if rogue wave added to dataset, choose non rogue window at random (might be better to do this sometime later, may be very slow)

figure out how to add to parquet datafile

Add functionality to iterate through text file

Add additional checks

Fix spike detection
'''

import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mpldts
import calendar
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Helper functions
def on_key(event):
    if event.key == 'q':
        plt.close()  # Close the figure window

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
    return len(zero_crossings), np.mean(np.diff(zero_crossings)) / sample_rate if len(zero_crossings) > 1 else 0

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
    N_z, T_z = calculate_zero_upcrossings(displacement_data, sample_rate)
    if T_z == 0:  # Prevent division by zero
        return None, None
    S_y = (4 * sigma / T_z) * np.sqrt(2 * np.log(N_z))
    # Discard the block if the threshold is exceeded
    if should_discard_block(displacement_data, sample_rate, S_y):
        # print("Measurement discarded due to exceeding the rate of change threshold.")
        return None, None

    troughs,crests = find_troughs_and_crests(displacement_data)
    wave_heights_info = calculate_wave_heights(displacement_data, troughs, crests)

    wave_heights = [info[2] for info in wave_heights_info]
    if not wave_heights:
        return None, None # No waves detected
    
    # Hs calculation: 4 times the standard deviation of the sea surface elevation
    # Hs = 4 * np.std(displacement_data)
    Hs = 4*np.std(displacement_data[np.abs(displacement_data) < 20.47])

    # Identify rogue waves
    for trough, crest, height in wave_heights_info:
        if height / Hs > 2:
            # Rogue wave detected
            # Calculate the number of samples for the normalization TODO: HARDCODE THIS
            pre_samples = int(25*60*sample_rate)
            post_samples = int(25*60*sample_rate)

            # Find index for normalization
            rogue_wave_index = crest

            # Calculate the start and end indices for the normalized window
            start_index = max(0, crest - pre_samples)
            end_index = min(len(displacement_data), crest + post_samples)

            # Extract normalized segment
            normalized_segment = displacement_data[start_index:end_index]

            # # Plotting
            # plt.figure(figsize=(20, 4))
            # plt.plot(normalized_segment, label='Displacement')
            
            # # Adjust indices for the trough and crest within the normalized segment
            # trough_index_within_segment = trough - start_index
            # crest_index_within_segment = crest - start_index
            
            # plt.plot(trough_index_within_segment, normalized_segment[trough_index_within_segment], 'ro', label='Trough')
            # plt.plot(crest_index_within_segment, normalized_segment[crest_index_within_segment], 'ro', label='Crest')
            
            # # Annotate Hs and H on the plot
            # plt.annotate(f'Hs = {Hs:.2f}m', xy=(0.05, 0.95), xycoords='axes fraction', verticalalignment='top')
            # plt.annotate(f'H = {height:.2f}m', xy=(crest_index_within_segment, normalized_segment[crest_index_within_segment]), 
            #              textcoords="offset points", xytext=(-10,10), ha='center', arrowprops=dict(arrowstyle="->", color='green'))
            
            # plt.title(f'Rogue Wave Detected - Normalized Segment')
            # plt.xlabel('Sample Index')
            # plt.ylabel('Displacement (m)')
            # plt.legend()
            # plt.gcf().canvas.mpl_connect('key_press_event', on_key)
            # plt.show()
            
            global rogue_count
            rogue_count += 1

            return normalized_segment, rogue_wave_index
        
    return None, None # No rogue wave detected

# load data
stn = '132'
dataset = 'archive'
deployment_max = '16'
# start_date = '02-09-06 15:00' # MM/DD/YYYY HH:MM
qc_level = 2 # TODO
rogue_count = 0

# main loop
for i in range(1, 17):
    deploy = f"{i:02}"

    # Archive
    data_url = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/' + stn + 'p1/' + stn + 'p1_d' + deploy + '.nc'

    nc = netCDF4.Dataset(data_url)
    # Turn off auto masking
    nc.set_auto_mask(False)

    z_displacement = nc.variables['xyzZDisplacement'][:] # Vertical displacement
    sample_rate = nc.variables['xyzSampleRate'][:].item() # Hz

    # Test implementation
    offset = 25*60*sample_rate
    block_duration = 30 # Minutes

    block_samples = int(block_duration*60*sample_rate)

    # Filter the displacement data based on the max sensor height condition
    filtered_displacement = z_displacement[np.abs(z_displacement) < 20.47]

    # Calculate the standard deviation for the filtered data
    sigma = np.std(filtered_displacement)


    for start_index in range(int(offset,), len(z_displacement), block_samples):
        end_index = start_index + block_samples
        # Ensure not to exceed the array bounds
        if end_index > len(z_displacement):
            break

        # Extract the current block of data
        current_block = z_displacement[start_index:end_index]

        # Detect rogue wave in current block
        # DEBUG:
        # Find crests
        crests, _ = find_peaks(current_block)

        # Find troughs by inverting the displacement data
        troughs, _ = find_peaks(-current_block)


        normalized_segment, rogue_wave_index = detect_rogue_wave(current_block, sample_rate, sigma)

        # if normalized_segment is not None:
        #     # Plot
        #     plt.figure(figsize=(10, 6))
        #     plt.plot(normalized_segment)
        #     plt.title(f'Rogue Wave Detected - Segment Centered at Index {rogue_wave_index}')
        #     plt.xlabel('Sample Index')
        #     plt.ylabel('Displacement (m)')
        #     plt.show()
    print(f"{rogue_count} rogue waves detected")
    print(f"deployment {i} finished")

print('done')
print(rogue_count)