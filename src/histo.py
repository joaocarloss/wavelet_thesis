import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

#%% Read frequency
file_path = r'C:\Pesquisa\project_wavelet_2024_master\wavelet_master\freq_new' # Frequency Path
date_name = Path(file_path).glob('*.txt')
list_data = []
for files_name in date_name:
    df_temp = pd.read_csv(files_name, sep = '	')
    list_data.append(df_temp)
    
data = pd.concat(list_data, ignore_index=True)

# Load data from the file
# file_path = r'C:\Pesquisa\wavelet_thesis\data_frequency.txt'
# data = pd.read_csv(file_path, sep='\t')

data.to_csv('C:\\Pesquisa\\wavelet_thesis\\main_ulf_test.csv', index = False)

# Convert frequency from Hz to mHz
data['freq(Hz)'] *= 1000

# Drop NaN values from the frequency column
data.dropna(subset=['freq(Hz)'], inplace=True)

# data_filtered = data[(data['freq(Hz)'] >= 5) & (data['freq(Hz)'] <= max(data['freq(Hz)']))]

# data_filtered = data[(data['freq(Hz)'] >= 5) & (data['freq(Hz)'] <= 50)]
data_filtered = data[(data['freq(Hz)'] >= 5)]



# data_5 = data[(data['freq(Hz)'] >= 5) & (data['freq(Hz)'] < 10)]
# data_15 = data[(data['freq(Hz)'] >= 15) & (data['freq(Hz)'] < 25)]
# data_25 = data[(data['freq(Hz)'] >= 25) & (data['freq(Hz)'] < 35)]
# data_35 = data[(data['freq(Hz)'] >= 35) & (data['freq(Hz)'] < 45)]
# data_45 = data[(data['freq(Hz)'] >= 45) & (data['freq(Hz)'] < 55)]
# data_55 = data[(data['freq(Hz)'] >= 55) & (data['freq(Hz)'] < 65)]
# data_65 = data[(data['freq(Hz)'] >= 65) & (data['freq(Hz)'] < 75)]
# data_75 = data[(data['freq(Hz)'] >= 75) & (data['freq(Hz)'] < 85)]
# data_85 = data[(data['freq(Hz)'] >= 85) & (data['freq(Hz)'] < 95)]
# data_95 = data[(data['freq(Hz)'] >= 95) & (data['freq(Hz)'] < 105)]
# data_105 = data[(data['freq(Hz)'] >= 105) & (data['freq(Hz)'] < 115)]
# data_115 = data[(data['freq(Hz)'] >= 115) & (data['freq(Hz)'] < 125)]
# data_125 = data[(data['freq(Hz)'] >= 125) & (data['freq(Hz)'] < 135)]

# ulf_data_min_5 = ulf_data_min[(ulf_data_min['freq(Hz)'] >= 5) & (ulf_data_min['freq(Hz)'] < 10)]
# ulf_data_min_15 = ulf_data_min[(ulf_data_min['freq(Hz)'] >= 15) & (ulf_data_min['freq(Hz)'] < 25)]
# ulf_data_min_25 = ulf_data_min[(ulf_data_min['freq(Hz)'] >= 25) & (ulf_data_min['freq(Hz)'] < 35)]
# ulf_data_min_35 = ulf_data_min[(ulf_data_min['freq(Hz)'] >= 35) & (ulf_data_min['freq(Hz)'] < 45)]
# ulf_data_min_45 = ulf_data_min[(ulf_data_min['freq(Hz)'] >= 45) & (ulf_data_min['freq(Hz)'] < 55)]
# ulf_data_min_55 = ulf_data_min[(ulf_data_min['freq(Hz)'] >= 55) & (ulf_data_min['freq(Hz)'] < 65)]
# ulf_data_min_65 = ulf_data_min[(ulf_data_min['freq(Hz)'] >= 65) & (ulf_data_min['freq(Hz)'] < 75)]
# ulf_data_min_75 = ulf_data_min[(ulf_data_min['freq(Hz)'] >= 75) & (ulf_data_min['freq(Hz)'] < 85)]
# ulf_data_min_85 = ulf_data_min[(ulf_data_min['freq(Hz)'] >= 85) & (ulf_data_min['freq(Hz)'] < 95)]
# ulf_data_min_95 = ulf_data_min[(ulf_data_min['freq(Hz)'] >= 95) & (ulf_data_min['freq(Hz)'] < 105)]
# ulf_data_min_105 = ulf_data_min[(ulf_data_min['freq(Hz)'] >= 105) & (ulf_data_min['freq(Hz)'] < 115)]
# ulf_data_min_115 = ulf_data_min[(ulf_data_min['freq(Hz)'] >= 115) & (ulf_data_min['freq(Hz)'] < 125)]
# ulf_data_min_125 = ulf_data_min[(ulf_data_min['freq(Hz)'] >= 125) & (ulf_data_min['freq(Hz)'] < 135)]

# data_50 = data[(data['freq(Hz)'] >= 55) & (data['freq(Hz)'] < 60)]


# data_125 = data[(data['freq(Hz)'] >= 105) & (data['freq(Hz)'] < 125)]

# data_125.to_csv(r"C:\Users\jcnet\Downloads\105_125_frq.txt", index = False,)

#%% Plotting the histogram
def plot_histo(data):

    # bin_width = 2
    # bins = np.arange(5,55,5)
    # bins = [5,10,15,20,25,30,35,40,45,50] # bin adriane compare
    # bins = [5,10,20,30,40,50,60,70,80,90,100,110,120,130] # Mars Adriane
    # bins = [5,15,25,35,45,55, 65, 75, 85,95,105,105,115,125]
    bins = np.arange(5,data['freq(Hz)'].max() + 10,10) # default
    
    # bins = np.arange(5, max(data['freq(Hz)']) + bin_width, bin_width)
    
    counts_freq = len(data)
    min_bin = min(bins)
    max_bin = max(bins)

    # Plotting the histogram
    plt.figure()
    # plt.style.use('_mpl-gallery')
    
    n, bins, patches = plt.hist(data['freq(Hz)'], bins=bins, weights=(np.ones(len(data)) / len(data)) * 100, edgecolor='#231f20ff',  linewidth=0.9)
    
    for patch, percentage in zip(patches, n):
        patch.set_facecolor('white')
        plt.text(patch.get_x() + patch.get_width() / 2, patch.get_height() + 0.5, f'{percentage:.1f}', ha='center',)
    
    plt.xlabel('Frequency (mHz)',)
    plt.ylabel(r'Percentage ($\%$)', )
    # plt.title('')
   

    plt.title(f' 2013 - ({counts_freq} total)')

    plt.xticks(bins, )
    plt.yticks()
    
    axs = plt.gca()

    axs.yaxis.set_major_locator(ticker.MultipleLocator(5))
    axs.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.ylim(0, 25)
    
    plt.tight_layout()
    plt.grid(False)
    plt.show()
    
plot_histo(data = data_filtered)

#%% Plotting the histograma V2.0
def plot_histo_2(data):
    # plt.rcParams['text.usetex'] = True

    bins_list = [[5,10,20,30,40,50], [5,10,15,20,25,30,35,40,45,50]]
    counts_freq = len(data)
    
    # plt.style.use('_mpl-gallery')
    plt.style.use('default')
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16,5))
    
    for ax, bins_ in zip(axs, bins_list):
        n, bins, patches = ax.hist(data['freq(Hz)'], bins=bins_, weights=(np.ones(len(data)) / len(data)) * 100, edgecolor='#231f20ff', linewidth=0.8)
        
        for patch, percentage in zip(patches, n):
            ax.text(patch.get_x() + patch.get_width() / 2, patch.get_height() + 0.1, f'{percentage:.1f}%', ha='center', va='bottom')
            # patch.set_facecolor('white')

        ax.set_xlabel('Frequency (mHz)')
        ax.set_ylabel('Percentage (%)')
        # ax.set_title(f'IC freq from {min(bins_)} to {max(bins_)} ({counts_freq} total)')
        ax.set_xticks(bins)
        # ax.grid(True)

    plt.subplots_adjust(top=0.945,
                        bottom=0.085,
                        left=0.035,
                        right=0.99,
                        hspace=0.4,
                        wspace=0.09) # Adjust the space between subplots if needed
    plt.show()

# plot_histo_2(data = data_filtered)
    
# data.to_csv(r'C:\Pesquisa\wavelet_thesis\ulf.csv', index = False)

