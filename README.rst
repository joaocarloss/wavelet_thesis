Wavelet Analysis Project
========================

Overview
--------
This repository is dedicated to storing the catalog list of VEX and the Morlet wavelet-based code for analyzing data from instruments on the VEX probe.

Problem Statement
-----------------
ULF waves are considered as an essential factor in the magnetospheric physics. Those waves carry information about instabilities, free energy configuration contained in the plasma or in the obstacles to the flow. Thus, the project addresses the challenge of identifying the main ULF waves in Venus magnetosheath from datasets provided.

Prerequisites
-------------
- Python version 3.x or higher.

Installation and Setup
----------------------
No specific installation process is required. The project runs directly from the source files. To get started, open the main file (``wavelet_main.py``) and run the code.

Usage
-----
Run the ``wavelet_main.py`` script to initiate the analysis. This script integrates all the functionalities needed for the wavelet analysis.

Features
--------
- **Catalog list**:  The boundaries crossings from Neto until 20011-2014.
- **Wavelet Analysis**: Using the Morlet wavelet to analyze the data.
- **ULF Wave Identification**: Click on the GW Spectrum plot to identify the main ULF waves.
- **Data Export**: Store the frequency of identified waves to CSV files for further analysis.
- **Wavelet Analysis**: Using the Morlet wavelet to analyze the data.
- **ULF Wave Identification**: Click on the GW Spectrum plot to identify the main ULF waves.
- **Data Export**: Store the frequency of identified waves to CSV files for further analysis.

Modules Interaction
-------------------
- `wavelet_main.py` : Main code to run
- `waveletFunctions.py`: Function to apply the wavelet transformation to the data.
- `plot_wavelet.py`: Handles the plotting of the wavelet for user interaction and ULF wave selection.
Data Format
-----------
Data files within the ``data_density`` folder must adhere to the following format:
- Single column data.
- No header.
- No index.

Contributing
------------
