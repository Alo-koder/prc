# Alek's Codebase

Hello! Have fun using my code :)


## Purpose

This is the code I used throughout my master's project
on [Phase Response Curves of an Electrochemical Silicon Oscillator](T:/Team/_Literatur/Theses/Silicon%20Oscillations/Master%20Alek%20Szewczyk%20(extern,%202024).pdf).

## Software and Packages

The easiest way to get all you need to run the program is via Anaconda.
Assuming you have it installed, you can simpy run the following commands
from this folder:

``` bash
conda env create f- conda_env.yaml
conda activate prc
pip install eclabfiles
```
The package eclabfiles isn't available on Anaconda,
therefore it has to be installed from pip.
Eclabfiles isn't necessary for experiments made after July 2024.

### Jupyter

This program bases on Jupyter notebooks.
Spyder's support for Jupyter is currently very limited.
For best experience use VSCode or [Jupyter Notebook / Jupyter Lab](https://jupyter.org/install).



## What to Expect

The codebase consists mainly of Jupyter notebooks and vanilla python scripts.
The notebooks are meant to be an interface –
ideally, you shouldn't need to interact with `.py` files at all
(except for [`config.py`](config.py)).
The `.py` files are collections of data analysis methods that are
used in notebooks.


## Usage

### Config Files

When analysing a new measurement, you should start by writing a `.yaml` config file
with experimental metadata.
A [commented example](properties_example.yaml) is provided for one of my measurements.

### [Create Pickle](create_pickle.ipynb)

The raw data from different measurement devices is first synchronised and combined
into a single [pickle](https://docs.python.org/3/library/pickle.html) file.

### [PRC Analysis](prc_analysis.ipynb)

This module reads previously prepared, single data file and calculates the PRC.
PRCs can be plotted immediately and/or saved as `.csv` files.
