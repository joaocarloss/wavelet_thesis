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
import pycwt
import datetime
import sys
from pathlib import Path
import re
import os
import json
from io import StringIO

from waveletFunctions import wavelet, wave_signif
from plot_wavelet import plot_wavelet

# Python wavelet software provided by Evgeniya Predybaylo
# based on Torrence and Compo (1998) and is available at URL: "http://atoc.colorado.edu/research/wavelets/"

# %% Load function


def read_all_dat(path):
    """Reads all .dat files from a given directory and normalizes the data.

    Created on Fri Feb 16 10:37:38 2024

    The data files must have 'data' in their name, for example, 'iVEX_SHEATH_S_EX_20080825_054301.dat'.
    The files must also contain a date in the 'YYYYMMDD' format.

    Args:
        path (str): Path to the folder containing .dat files.

    Returns:
        density (list): List of .dat files.
        date (list): List of the files' dates.e

    """

    # path = 'C:\\Pesquisa\\project_wavelet_2024_master\\magnetoshealt\\'

    path_dat_file = Path(path).glob("*.dat")

    date_files, density = [], []
    for f in path_dat_file:
        # save the date

        date = re.findall("\d{8}", f.name)[-1]
        if date in date_files:
            date_files.append(date + "_2")
        else:
            date_files.append(date)
        # save the .dat file

        dat = np.loadtxt(f)
        dat = dat - np.mean(dat)  # Normalize
        density.append(dat)
    return density, date_files


def read_one_dat(path):
    """Read one .dat file from the specified path.

    Created on Fri Feb 16 10:00:38 2024

    Args:
        path (str): Path to the folder containing the .dat file.

    Returns:
        density (array): Array representation of the .dat file.
        date (str): The file's date.
    """
    density = np.loadtxt(path)
    density = density - np.mean(density)  # Normalize

    return density


def read_json_file(file):
    """Read json that include density and time series period"""

    date_files, density, deltatime = [], [], []

    with open(f"{file}", "r") as fh:
        data = json.load(fh)
    for key, value in data.items():
        data = np.loadtxt(StringIO(value[0]))
        data = data - np.mean(data)
        time_temp = value[1]

        date_files.append(key)
        density.append(data)
        deltatime.append(time_temp)
    return density, date_files, deltatime


def ar1(x):
    """
    Allen and Smith autoregressive lag-1 autocorrelation coefficient.
    In an AR(1) model

        x(t) - <x> = \gamma(x(t-1) - <x>) + \alpha z(t) ,

    where <x> is the process mean, \gamma and \alpha are process
    parameters and z(t) is a Gaussian unit-variance white noise.

    Parameters
    ----------
    x : numpy.ndarray, list
        Univariate time series

    Returns
    -------
    g : float
        Estimate of the lag-one autocorrelation.
    a : float
        Estimate of the noise variance [var(x) ~= a**2/(1-g**2)]
    mu2 : float
        Estimated square on the mean of a finite segment of AR(1)
        noise, mormalized by the process variance.

    References
    ----------
    [1] Allen, M. R. and Smith, L. A. Monte Carlo SSA: detecting
        irregular oscillations in the presence of colored noise.
        *Journal of Climate*, **1996**, 9(12), 3373-3404.
        <http://dx.doi.org/10.1175/1520-0442(1996)009<3373:MCSDIO>2.0.CO;2>
    [2] http://www.madsci.org/posts/archives/may97/864012045.Eg.r.html

    """
    x = np.asarray(x)
    N = x.size
    xm = x.mean()
    x = x - xm

    # Estimates the lag zero and one covariance

    c0 = x.transpose().dot(x) / N
    c1 = x[0 : N - 1].transpose().dot(x[1:N]) / (N - 1)

    # According to A. Grinsteds' substitutions

    B = -c1 * N - c0 * N**2 - 2 * c0 + 2 * c1 - c1 * N**2 + c0 * N
    A = c0 * N**2
    C = N * (c0 + c1 * N - c1)
    D = B**2 - 4 * A * C

    if D > 0:
        g = (-B - D**0.5) / (2 * A)
    else:
        raise Warning(
            "Cannot place an upperbound on the unbiased AR(1). "
            "Series is too short or trend is to large."
        )
    # According to Allen & Smith (1996), footnote 4

    mu2 = -1 / N + (2 / N**2) * (
        (N - g**N) / (1 - g) - g * (1 - g ** (N - 1)) / (1 - g) ** 2
    )
    c0t = c0 / (1 - mu2)
    a = ((1 - g**2) * c0t) ** 0.5

    return g, a, mu2


# %% Load Files


density_all, date = read_all_dat(Path.cwd().parent / "data"/ "data_density")
# density_all, date = read_all_dat('C:\\Pesquisa\\project_wavelet_2024_master\\magnetoshealt\\')
# density, variance = read_one_dat(path ='C:\\Pesquisa\\project_wavelet_2024_master\\magnetoshealt\\iVEX_SHEATH_S_EX_20080825_054301.dat')
# density_all, date, deltatime = read_json_file('density_time_json')

# %% # Wavelet Transform

# Integrating the wavelet transform code into the process_data function

# normalize by standard deviation (not necessary, but makes it easier
# to compare with plot on Interactive Wavelet page, at
# "http://paos.colorado.edu/research/wavelets/plot/"


def process_data(density):
    """Process the density data using wavelet transform.

    Args:
        density (array_like): Density data.

    Returns:
        tuple: Contains time, period, sig95, coi, global_ws, global_signif, and T.
    """
    # Data processing and wavelet transformation

    n = len(density)
    # T = (n * 4)/60 # obtaing the period of time series in minutes

    T = 14.24
    time = np.linspace(0, T, n)  # timestep of each data point
    step = 4  # maximum time resolution of one energy sweep (128 steps) per 4 seconds
    dt = step / 60  # 4s -> 1/15 min (0.066min)

    pad = 1
    dj = 0.125 / 10  # Nitidez
    # dj = 1/12
    # dj = 0.125/2 (default matlab)

    s0 = dt  # this says start at a scale of 1/4
    # s0 = 0.6*dt (default matlab)

    j1 = 5 / dj
    try:
        lag1, _, _ = ar1(
            density
        )  # Install pycwt to use Allen and Smith autoregressive lag-1 autocorrelation coefficient.
    except:
        # Look the warning in the ar1

        lag1 = 0.72  # Lag-1 autocorrelation for red noise (default matlab)
        print(
            "Cannot place an upperbound on the unbiased AR(1).\n"
            "Series is too short or trend is to large.\n\n"
            "Autocorrelation given as lag1 = 0.72"
        )
    mother = "MORLET"

    # Wavelet transform

    wave, period, scale, coi = wavelet(density, dt, pad, dj, s0, j1, mother)
    power = (np.abs(wave)) ** 2
    global_ws = np.sum(power, axis=1) / n

    # Significance levels

    variance = np.std(density, ddof=1) ** 2
    signif, fft_theor = wave_signif(
        ([variance]), dt=dt, sigtest=0, scale=scale, lag1=lag1, mother=mother
    )
    sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])
    sig95 = power / sig95

    # Global wavelet spectrum & significance levels

    dof = n - scale
    global_signif, _ = wave_signif(
        variance, dt=dt, scale=scale, sigtest=1, lag1=lag1, dof=dof, mother=mother
    )

    return (
        density,
        time,
        power,
        period,
        variance,
        sig95,
        coi,
        global_ws,
        fft_theor,
        global_signif,
        T,
    )


# Select the by data (comment below if using read_one_dat)
# x = date.index('20071216_2') #select a day

x = date.index("20071227_2")  # select a day
density = density_all[x]

(
    density,
    time,
    power,
    period,
    variance,
    sig95,
    coi,
    global_ws,
    fft_theor,
    global_signif,
    T,
) = process_data(density=density)

# %% Plot the wavelet


save = Path.cwd()

selected_freq = plot_wavelet(
    density=density,
    time=time,
    power=power,
    period=period,
    variance=variance,
    sig95=sig95,
    coi=coi,
    fft_theor=fft_theor,
    global_ws=global_ws,
    global_signif=global_signif,
    T=T,
    date=date[x],
    enable_click=True,
    output_save=save,
)
