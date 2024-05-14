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

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.sans-serif": "cm",
# })

#%%

def plot_wavelet(sst, time, power, period, variance, sig95, coi, global_ws, fft_theor, global_signif, T, date, enable_click, output_save = Path.cwd()):
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
    plt.plot(time, sst, color='k', lw = 1)
    plt.xlim(xlim[:])
    plt.xlabel('Time (min)',  fontsize = 12)
    plt.ylabel('cm$^3$',  fontsize = 12)
    plt.title('a) Electron Density', fontsize = 12)
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelsize=9)
    ax.yaxis.set_tick_params(labelsize=9)
    plt.grid()

    # --- Plot wavelet spectrum (WT)
    plt3 = plt.subplot(gs[1, 0:3])
    plt.pcolormesh(time, period, power, cmap='jet', rasterized=True)
    plt.xlabel('Time (min)',  fontsize = 12)
    plt.ylabel('Period (min)',  fontsize = 12)
    plt.title(f'b) Wavelet Power Spectrum, {date_ext}', fontsize = 12)
    plt.xlim(xlim[:])
    # 95# significance contour, levels at -99 (fake) and 1 (95# signif)
    plt.contour(time, period, sig95, [-99, 1], colors='k', linewidths = 2)
    # cone-of-influence, anything "below" is dubious
    plt.fill_between(time, coi * 0 + period[-1], coi, facecolor="none", edgecolor="#00000040", hatch='x')
    plt.plot(time, coi, 'k', lw = 1)
    # format y-scale
    plt3.set_yscale('log', base=2, subs=None)
    plt.ylim([np.min(period), np.max(period)])
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_tick_params(labelsize=9)
    ax.yaxis.set_tick_params(labelsize=9)
    plt3.ticklabel_format(axis='y', style='plain')
    plt3.invert_yaxis()
    # plt.grid()

     # --- Plot global wavelet spectrum (GWS)
    plt4 = plt.subplot(gs[1, -1])
    plt.plot(global_ws, period, color='k', lw = 1)
    # plt.plot(global_signif, period, '--', color='k', lw = 1) # cone default
    # plt.plot(variance * fft_theor, period, '--', color='k', lw = 1) # fft cone
    plt.xlabel('Power (Amplitude$^2$)', fontsize = 12)
    plt.title('c) Global Wavelet Spectrum', fontsize = 12)
    plt.xlim([0, 1.25 * np.max(global_ws)])
    # format y-scale
    plt4.set_yscale('log', base=2, subs=None)
    plt.ylim([np.min(period), np.max(period)])
    # ax = plt.gca().yaxis
    # ax.set_major_formatter(ticker.ScalarFormatter())
    
    ax = plt.gca()
    # ax.xaxis.get_major_formatter().set_useOffset(False)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    major_ticks = np.arange(np.min(np.log2(period)), np.max(np.log2(period)), 1)    
    minor_ticks = 2**(major_ticks + 0.4)
    ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
    ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, pos: ''))
    ax.xaxis.set_tick_params(labelsize=9)
    ax.yaxis.set_tick_params(labelsize=9)
    plt4.ticklabel_format(axis='y', style='plain')
    
    # write annnote
    # ax.plot(39500, 11, 'o', color = 'black')
    ax.annotate('$27.2$ mHz', xy=(4.5, 0.62), xytext = (4.5, 0.25),
                  ha='center', va='center', fontsize = 12, 
                  arrowprops=dict(arrowstyle= '->', lw=1.35, facecolor='k'),
                  bbox=dict(fc="white", ec="none", pad=0.2))

    ax.annotate('$12.4$ mHz', xy=(15.5, 1.34), xytext = (15.5, 0.4),
                  ha='center', va='center', fontsize = 12, 
                  arrowprops=dict(arrowstyle= '->', lw=1.35, facecolor='k'),
                  bbox=dict(fc="white", ec="none", pad=0.2))

    # facecolor= black
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
                    directory = output_save /'saves'
                    # df_click.to_csv(f'{directory}\{name_file}.txt', sep='\t', index=False)
                    # print(f'Data {name_file}.txt save in: as {directory} and disconnected')
                    selected_points.clear()
                    plt.close()
        
        # Create a cursor grid
        cursor = Cursor(plt4, useblit=True, color='red', linewidth=0.8, linestyle='--')
        # Connect the click event to the onclick function
        fig.canvas.mpl_connect('button_press_event', onclick)

    plt.tight_layout()
    plt.show()
    # save_fig = output_save /'good_plot_wavelet'
    # plt.savefig(f"{save_fig}\{date}.pdf", dpi =300)

    return selected_points