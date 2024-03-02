import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rogue = pd.read_parquet(r"Rogue Wave Data\CORRECTED Rogue Wave Data\rogue_wave_data_station_132.parquet", engine="pyarrow")
#non_rogue = pd.read_parquet(r"data\non_rogue\non_rogue_wave_data_station_143_01.parquet", engine="pyarrow")


i = 0
with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision',3,):
    i = i + 1
    
#print(non_rogue.shape)

'''
Given Rogue wave dataset has:
    4 Rogue rogue
    7684 Time points
    30 days time period
'''

#test = rogue.iloc[:,3].to_numpy(dtype=int)
#print(sum(test))

#test = 0
#test = non_rogue.iloc[:,3].to_numpy(dtype=int)
#print(sum(test))

#print(non_rogue.shape)
rogue.head()
