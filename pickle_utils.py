import pandas as pd
import eclabfiles as ecf
import numpy as np
from datetime import datetime
from scipy.optimize import minimize


def read_emsi(emsi_filename: str) -> tuple[pd.DataFrame, float, float]:
    '''
    Read the emsi `.dat` files.

    Parameters
    ----------
    emsi_filename   : str

    Returns
    -------
    emsidf          : pd.DataFrame
    start_time      : float
        UTS timestamp at which the data in this file starts.
    end_time        : float
        UTS time of file end.
    '''
    emsidf = pd.read_csv(emsi_filename, sep='\t', skiprows = 5)
    new_col_names = { # original column names are super annoying to type
        'Time [s]'  : 't',
        'U [V]'     : 'U',
        'I [A]'     : 'I',
        'Ill [V]'   : 'light',  # voltage on the shutter;
        'EMSI [%]'  : 'emsi',   # high voltage means loss light goes through
    }
    emsidf = emsidf.rename(columns = new_col_names)
    emsidf['U'] = -emsidf['U']  # For some reason the setup reads electricity backwards
    emsidf['I'] = -emsidf['I']

    # Converting start time to uts (unix time stamp)
    with open(emsi_filename, 'r') as file:
        first_line = file.readline()
    date_str = first_line.split(': ')[1][:-1]
    date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    start_time = date_obj.timestamp()
    end_time = start_time+emsidf.t.iloc[-1]

    return emsidf, start_time, end_time


def read_ecf(ecf_filename: str) -> tuple[pd.DataFrame, float]:
    '''
    LEGACY Import the potentiostat data from a `.mpr` file.

    This will not work with the newest version of the ECLab software.
    Export the files to `.txt` and use `read_ecf_text()` instead.

    Parameters
    ----------
    ecf_filename    : str
        The `.mpr` measurement file

    Returns
    -------
    ecfdf           : pd.DataFrame
        A pandas data frame with all useful potentiostat data
    start_time      : float
        UTS timestamp of the experiment's start
    '''    
    try:
        import eclabfiles as ecf
    except ImportError:
        print("eclabfiles not available")
    ecfdf = ecf.to_df(ecf_filename)
    start_time = ecfdf.uts[0]

    new_col_names = {
        'time'  : 't',
        'Ewe'   : 'U',
        '<I>'   : 'I',
    }

    ecfdf = ecfdf[new_col_names.keys()]
    ecfdf = ecfdf.rename(columns = new_col_names)

    return ecfdf, start_time


def read_ecf_text(ecf_text_filename: str) -> tuple[pd.DataFrame, float]:
    '''
    Import the potentiostat data from a text file.

    Parameters
    ----------
    ecf_filename    : str
        The exported `.txt` potentiostat data

    Returns
    -------
    ecfdf           : pd.DataFrame
        A pandas data frame with all useful potentiostat data
    start_time      : float
        UTS timestamp of the experiment's start
    '''
    with open(ecf_text_filename) as file:
        lines = file.readlines()
    Nb_header_lines = int(lines[1].split(' : ')[-1])
    header_lines = lines[:Nb_header_lines-1]

    ecf_start_string = [s.split(' : ')[-1][:-1] for s in header_lines if 'Acquisition started on' in s][0]
    ecf_start_obj = datetime.strptime(ecf_start_string, "%m/%d/%Y %H:%M:%S.%f")
    ecf_start = ecf_start_obj.timestamp()

    # I assume you export time, voltage and current in that order.
    # Otherwise change the `names` parameter below accordingly.
    ecfdf = pd.read_csv(ecf_text_filename, decimal=',', names=['t', 'U', 'I'],
                        sep='\t', skiprows=Nb_header_lines, encoding='ANSI')

    return ecfdf, ecf_start
    

def read_photodiode(ph_filename: str) -> tuple[pd.DataFrame, float]:
    '''
    Read the photodiode data.

    Parameters
    ----------
    ph_filename : str

    Returns
    -------
    phdf        : pd.DataFrame
    ph_start    : float
        UTS timestamp of the beginnig of the photodiode reading.
    '''
    with open(ph_filename) as file:
        ph_start_str = file.readline().split(sep='\t')[-1][:-1]
    date_obj = datetime.strptime(ph_start_str, "%Y-%m-%d %H:%M:%S")
    ph_start = date_obj.timestamp()

    phdf = pd.read_csv(ph_filename, sep='\t', decimal=',', skiprows=1, header=None)
    phdf.columns = ['t', 'photodiode']
    phdf.photodiode *= 1000 # convert illumination from W to mW
    phdf.t = phdf.t.astype(float)/1000  # by default the photodiode stores time in miliseconds;
    return phdf, ph_start               # we convert it to seconds



def optimise_emsi_start(ecfdf:pd.DataFrame, ecf_start:float, emsidf:pd.DataFrame, emsi_start:float) -> float:
    '''
    Finely adjust LabView recorded time to ensure sync with the potentiostat data.

    LabView is annoying and stores the experiment start only with 1s accuracy.
    Trusting this would cause a ~1s mismatch between `ecfdf` and `emsidf`.
    Instead, LabView current data is fitted to the potentiostat's current
    and the time shift that gives the best match is returned.

    Parameters
    ----------
    ecfdf       : pd.DataFrame
    ecf_start   : float (uts timestamp)
    emsidf      : pd.DataFrame
    emsi_start  : float (uts timestamp)

    Returns
    -------
    shift       : float (uts timestamp)
        `emsidf` time is lagging by [[shift]] seconds compared to `ecfdf`.
    '''
    
    roi_start = max(ecf_start, emsi_start) + 600
    roi_end = roi_start + 6600
    ecf_useful = ecfdf.loc[(ecfdf.t > roi_start-ecf_start) & (ecfdf.t < roi_end-ecf_start)]
    emsi_useful = emsidf.loc[(emsidf.t > roi_start-10-emsi_start) & (emsidf.t < roi_end+10-emsi_start)]
    emsi_useful.loc[:, 't'] = emsi_useful.t+emsi_start-roi_start
    ecf_useful.loc[:, 't'] = ecf_useful.t+ecf_start-roi_start

    def fit(shift: float) -> float:
        loss = np.sum(np.square(emsi_useful.I - np.interp(emsi_useful.t, ecf_useful.t+shift, ecf_useful.I)))
        return loss
    shift = minimize(fit, 1).x[0]
    print(f"{shift = }")
    return shift