"""
Created on Thu Feb 15 09:29:31 2024
@author: jcnet

Apply the Morlet wavelet and plot.
It is necessary waveletFunctions and plot_wavelet in the same directory.

    16/02/2024
Created a merge all .dat files
Added a new level colour map 
Created a save of click (frequency)

    23/02/2024
Created library plot_wavelet, thus you do not need to creat a plot

    24/02/2024
Implemented 'while plt.fignum_exists(fig.number)' to plot a sequence

"""

# %% Import libraries
import numpy as np
import pandas as pd
from waveletFunctions import wavelet, wave_signif
import pycwt 
import datetime
import sys
from pathlib import Path
import re
import os

from plot_wavelet import plot_wavelet
# 
__author__ = 'Evgeniya Predybaylo'


# WAVETEST Example Python script for WAVELET, using NINO3 SST dataset
#
# See "http://paos.colorado.edu/research/wavelets/"
# The Matlab code written January 1998 by C. Torrence is modified to Python by Evgeniya Predybaylo, December 2014
#
# Modified Oct 1999, changed Global Wavelet Spectrum (GWS) to be sideways,
#   changed all "log" to "log2", changed logarithmic axis on GWS to
#   a normal axis.
    

#%% Load function

def read_all_dat(path):
    
    """Reads all .dat files from a given directory and normalizes the data.
    
    Created on Fri Feb 16 10:37:38 2024
    
    The data files must have 'data' in their name, for example, 'iVEX_SHEATH_S_EX_20080825_054301.dat'. 
    The files must also contain a date in the 'YYYYMMDD' format.

    Args:
        path (str): Path to the folder containing .dat files.

    Returns:
        sst (list): List of .dat files.
        date (list): List of the files' dates.e
    
    """
   
    # path = 'C:\\Pesquisa\\project_wavelet_2024_master\\magnetoshealt\\'
    path_dat_file = Path(path).glob('*.dat')
    
    date_files, sst = [], []
    for f in path_dat_file:
        # save the date
        date = (re.findall('\d{8}', f.name)[-1])
        if date in date_files:
            date_files.append(date + "_2")
        else:
            date_files.append(date)
        
        # save the .dat file
        dat = np.loadtxt(f)
        dat = dat - np.mean(dat) # Normalize
        sst.append(dat)
        
    return sst, date_files


def read_one_dat(path):
    """Read one .dat file from the specified path.
    
    Created on Fri Feb 16 10:00:38 2024

    Args:
        path (str): Path to the folder containing the .dat file.

    Returns:
        sst (array): Array representation of the .dat file.
        date (str): The file's date.
    """
    sst = np.loadtxt(path)
    sst = sst - np.mean(sst) # Normalize
  
    return sst


#%% Load Files

sst_all, date = read_all_dat(Path.cwd()/'data')
# sst, variance = read_one_dat(path ='C:\\Pesquisa\\project_wavelet_2024_master\\magnetoshealt\\iVEX_SHEATH_S_EX_20080825_054301.dat')

#%% # Wavelet Transform

# Integrating the wavelet transform code into the process_data function

# normalize by standard deviation (not necessary, but makes it easier
# to compare with plot on Interactive Wavelet page, at
# "http://paos.colorado.edu/research/wavelets/plot/"

def process_data(sst):
    """Process the SST data using wavelet transform.

    Args:
        sst (array_like): Density data.

    Returns:
        tuple: Contains time, period, sig95, coi, global_ws, global_signif, and T.
    """
    # Data processing and wavelet transformation
    n = len(sst)
    dt = (4 / 60)  # in minutes
    T = n * dt # obtaing the period of time series in minutes
    time = np.linspace(0, T, n) # timestep of each data point
    
    
    pad = 1
    dj = 0.125 / 10
    s0 = dt
    j1 = 5 / dj
    # lag1 = 0.70 # Lag-1 autocorrelation for red noise (Default)
    lag1, _, _ = pycwt.ar1(sst)  # Install pycwt to use Allen and Smith autoregressive lag-1 autocorrelation coefficient.
    mother = 'MORLET'
    
    # Wavelet transform
    wave, period, scale, coi = wavelet(sst, dt, pad, dj, s0, j1, mother)
    power = (np.abs(wave)) ** 2
    global_ws = (np.sum(power, axis=1) / n)

    # Significance levels
    variance = np.std(sst, ddof=1) ** 2
    signif = wave_signif(([variance]), dt=dt, sigtest=0, scale=scale, lag1=lag1, mother=mother)
    sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])
    sig95 = power / sig95

    # Global wavelet spectrum & significance levels
    dof = n - scale
    global_signif = wave_signif(variance, dt=dt, scale=scale, sigtest=1, lag1=lag1, dof=dof, mother=mother)

    return sst, time, power, period, sig95, coi, global_ws, global_signif, T

# Select the by data (comment below if using read_one_dat)
x = date.index('20070210') #select
sst = sst_all[x]

sst, time, power, period, sig95, coi, global_ws, global_signif, T = process_data(sst = sst)

#%% Plot the wavelet

save_freq = Path.cwd() /'saves'

selected_freq = plot_wavelet(sst = sst, time = time, 
                             power = power, period = period, 
                             sig95 = sig95, coi = coi, 
                             global_ws = global_ws,global_signif = global_signif, 
                             T = T, date = date[x], 
                             enable_click=True, output_save = save_freq)