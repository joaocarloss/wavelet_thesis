# Wavelet Analysis Project

## Overview

This repository is dedicated to storing the catalog list of VEX and the Morlet wavelet-based code for analyzing data from instruments on the VEX probe.

## Problem Statement
ULF waves are considered as an essential factor in the magnetospheric physics. Those waves carry information about instabilities, free energy configuration contained in the plasma or in the obstacles to the flow. Thus, the project addresses the challenge of identifying the main ULF waves in Venus magnetosheath from datasets provided.

### Requirements

 Install `requirements.txt`

- Python (>= 3.x)
- Pandas, for data manipulation  
- NumPy, for all numerical algorithms
- Matplotlib, for static plotting and visualizations
- Scipy, for scientific computing 

## How do I use?

Run the `wavelet_main.py` script to initiate the analysis. This script integrates all the functionalities needed for the wavelet analysis.

## Features
- **Catalog list**:  The boundaries crossings from Neto until 20011-2014.
- **Wavelet Analysis**: Using the Morlet wavelet to analyze the data.
- **ULF Wave Identification**: Click on the GW Spectrum plot to identify the main ULF waves.
- **Data Export**: Store the frequency of identified waves to CSV files for further analysis.

## Modules Interaction

- `wavelet_main.py` : Main code to run
- `waveletFunctions.py`: Function to apply the wavelet transformation to the data.
- `plot_wavelet.py`: Handles the plotting of the wavelet for user interaction and ULF wave selection.

## Data Format
Data files within the `data_density` folder must adhere to the following format:
- Single column data.
- No header.
- No index.

