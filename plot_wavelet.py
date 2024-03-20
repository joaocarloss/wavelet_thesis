# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 13:36:40 2024

Created for facility the wavelet plotting

Modified on: 
    16/02/2024
    created a save of cliks, to run: 
        left button - select the frequency
        right button - stop selecting and save the clicks. 
        ps: be sure that, file .txt sep \tab and 2 columns = [date, freq]
        
    24/02/2024
    Implemented 'while plt.fignum_exists(fig.number)' to plot a sequence

@author: jcnet
"""
import numpy as np
import datetime
import pandas as pd
from pathlib import Path
from decimal import Decimal

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from matplotlib.widgets import Cursor


#%%

def plot_wavelet(sst, time, power, period, sig95, coi, global_ws, global_signif, T, date, enable_click, output_save = Path.cwd()):
    """Plot a wavelet transform visualization.

    Args:
        sst (array_like): The dataset.
        time (array_like): Time data.
        power (array_like): Wavelet power spectrum data.
        period (array_like): Period data for the wavelet.
        sig95 (array_like): 95% significance level data.
        coi (array_like): Cone of influence data.
        global_ws (array_like): Global wavelet spectrum data.
        global_signif (array_like): Global significance level data.
        T (float): Total time of the dataset.
        date (str): Date string in 'YYYYMMDD' format.
        enable_click (bool): If True, enables onclick functionality for interactive plot selection.

    Returns:
        list: List of selected points when enable_click is True, otherwise an empty list.
    """

    plt.style.use('seaborn-v0_8-paper')
    
    xlim = ([0, T])
    
    # --- Config the date
    # Check if the date ends with '_2' and remove it if it does
    date_str = date[:8] if date.endswith('_2') else date

    # Convert the date string to a datetime object
    date_new = datetime.datetime.strptime(date_str, '%Y%m%d')

    # Decide whether the date is 'inbound' or 'outbound'
    if date.endswith('_2'):
        date_ext = date_new.strftime('%Y %b %d (outbound)')
    else:
        date_ext = date_new.strftime('%Y %b %d (inbound)')
    
    # date_new = datetime.datetime.strptime(f'{date}','%Y%m%d')
    # date_ext =  date_new.strftime('%Y %b %d')
    date_save =  date_new.strftime('%Y%m%d')

    # --- Plot time series
    fig = plt.figure(figsize=(12, 7))
    gs = GridSpec(2, 4)
    plt.subplot(gs[0, 0:3])
    plt.plot(time, sst)
    plt.xlim(xlim[:])
    plt.xlabel('Time (min)')
    plt.ylabel('cm$^3$')
    plt.title('a) Density')
    plt.grid()

    # --- Plot wavelet spectrum (WT)
    plt3 = plt.subplot(gs[1, 0:3])
    plt.pcolormesh(time, period, power, cmap='jet')
    plt.xlabel('Time (min)')
    plt.ylabel('Period (min)')
    plt.title(f'b) Wavelet Power Spectrum, {date_ext}')
    plt.xlim(xlim[:])
    # 95# significance contour, levels at -99 (fake) and 1 (95# signif)
    plt.contour(time, period, sig95, [-99, 1], colors='k')
    # cone-of-influence, anything "below" is dubious
    plt.fill_between(time, coi * 0 + period[-1], coi, facecolor="none", edgecolor="#00000040", hatch='x')
    plt.plot(time, coi, 'k')
    # format y-scale
    plt3.set_yscale('log', base=2, subs=None)
    plt.ylim([np.min(period), np.max(period)])
    ax = plt.gca().yaxis
    ax.set_major_formatter(ticker.ScalarFormatter())
    plt3.ticklabel_format(axis='y', style='plain')
    plt3.invert_yaxis()
    # plt.grid()

     # --- Plot global wavelet spectrum (GWS)
    plt4 = plt.subplot(gs[1, -1])
    plt.plot(global_ws, period)
    plt.plot(global_signif, period, '--')
    plt.xlabel('Power (Amplitude$^2$)')
    plt.title('c) Global Wavelet Spectrum')
    plt.xlim([0, 1.25 * np.max(global_ws)])
    # format y-scale
    plt4.set_yscale('log', base=2, subs=None)
    plt.ylim([np.min(period), np.max(period)])
    ax = plt.gca().yaxis
    ax.set_major_formatter(ticker.ScalarFormatter())
    plt4.ticklabel_format(axis='y', style='plain')
    plt4.invert_yaxis()
    plt.grid()

    selected_points = []

    if enable_click:
        def onclick(event):
            if event.inaxes == plt4:
                if event.button == 1:  # Left mouse button
                    ix, iy = event.xdata, event.ydata
                    print(f"Point selected: {date} - min: {round(iy, 2)} min, y: {Decimal((1/(iy*60))):.3E} Hz")
                    selected_points.append([date, (1/(iy*60))]) # convert min to Hz
                elif event.button == 3:  # Right mouse button
                    name_file = f'{date}_freq'
                    df_click = pd.DataFrame(selected_points, columns=['date','freq'])
                    directory = output_save
                    df_click.to_csv(f'{directory}\{name_file}.txt', sep='\t', index=False)
                    print(f'Data {name_file}.txt save in: as {directory} and disconnected')
                    selected_points.clear()
                    plt.close()
        
        # Create a cursor grid
        cursor = Cursor(plt4, useblit=True, color='red', linewidth=0.8, linestyle='--')
        # Connect the click event to the onclick function
        fig.canvas.mpl_connect('button_press_event', onclick)

    plt.tight_layout()
    plt.show()

    return selected_points