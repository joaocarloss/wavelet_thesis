Wavelet Analysis Project
========================

Overview
--------
This project focuses on the wavelet analysis of data from instruments on the VEX probe using the Morlet wavelet. The primary goal is to identify the main Ultra Low Frequency (ULF) waves.

Problem Statement
-----------------
The project addresses the challenge of identifying the main ULF waves in the datasets provided.

Prerequisites
-------------
- Python version 3.x or higher.

Installation and Setup
----------------------
No specific installation process is required. The project runs directly from the source files. To get started, open the main file (``wavelet_master_v2.py``) and run the code.

Usage
-----
Run the ``wavelet_master_v2.py`` script to initiate the analysis. This script integrates all the functionalities needed for the wavelet analysis.

Features
--------
- **Wavelet Analysis**: Using the Morlet wavelet to analyze the data.
- **ULF Wave Identification**: Click on the GW Spectrum plot to identify the main ULF waves.
- **Data Export**: Store the frequency of identified waves to CSV files for further analysis.

Modules Interaction
-------------------
- ``waveletFunctions.py``: Applies the wavelet transformation to the data.
- ``plot_wavelet.py``: Handles the plotting of the wavelet for user interaction and ULF wave selection.

Data Format
-----------
Data files within the ``density_2007`` folder must adhere to the following format:
- Single column data.
- No header.
- No index.

Contributing
------------
