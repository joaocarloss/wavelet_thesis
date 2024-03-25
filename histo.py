import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# file_path = r'C:\Pesquisa\project_wavelet_2024_master\wavelet_master\freq_2007_new' # Frequency Path
# date_name = Path(file_path).glob('*.txt')
# list_data = []
# for files_name in date_name:
#     df_temp = pd.read_csv(files_name)
#     list_data.append(df_temp)
    
# data = pd.concat(list_data)

# Load data from the file
file_path = r'C:\Pesquisa\wavelet_thesis\data_frequency.txt'
data = pd.read_csv(file_path, sep='\t')

# Convert frequency from Hz to mHz
data['freq(Hz)'] *= 1000

# Drop NaN values from the frequency column
data.dropna(subset=['freq(Hz)'], inplace=True)

# Plotting the histogram
bin_width = 10
bins = np.arange(min(data['freq(Hz)']), max(data['freq(Hz)']) + bin_width, bin_width)

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.style.use('_mpl-gallery')

n, bins, patches = plt.hist(data['freq(Hz)'], bins=bins, weights=(np.ones(len(data)) / len(data)) * 100, edgecolor='#231f20ff',  linewidth=0.6)

for patch, percentage in zip(patches, n):
    plt.text(patch.get_x() + patch.get_width() / 2, patch.get_height() + 0.1, f'{percentage:.1f}%', ha='center')

plt.xlabel('Frequency (mHz)')
plt.ylabel('Percentage (%)')
plt.xticks(bins)
plt.grid(True)
plt.tight_layout()
plt.show()

