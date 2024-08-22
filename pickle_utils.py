import pandas as pd
import eclabfiles as ecf
import numpy as np
from datetime import datetime
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.ndimage import convolve1d, gaussian_filter1d

def read_emsi(emsi_filename: str) -> tuple[pd.DataFrame, float, float]:
    # Reading with pandas
    emsidf = pd.read_csv(emsi_filename, sep='\t', skiprows = 5)
    new_col_names = {
        'Time [s]'  : 't',
        'U [V]'     : 'U',
        'I [A]'     : 'I',
        'Ill [V]'   : 'light',
        'EMSI [%]'  : 'emsi',
    }
    emsidf = emsidf.rename(columns = new_col_names)
    emsidf['U'] = -emsidf['U']
    emsidf['I'] = -emsidf['I']

    # Converting start time to uts
    with open(emsi_filename, 'r') as file:
        first_line = file.readline()
    date_str = first_line.split(': ')[1][:-1]
    date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    start_time = date_obj.timestamp()
    end_time = start_time+emsidf.t.iloc[-1]

    return emsidf, start_time, end_time

def read_ecf(ecf_filename: str) -> tuple[pd.DataFrame, float]:

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

def read_ecf_text(ecf_text_filename: str):
    with open(ecf_text_filename) as file:
        lines = file.readlines()
    Nb_header_lines = int(lines[1].split(' : ')[-1])
    header_lines = lines[:Nb_header_lines-1]

    ecf_start_string = [s.split(' : ')[-1][:-1] for s in header_lines if 'Acquisition started on' in s][0]
    ecf_start_obj = datetime.strptime(ecf_start_string, "%m/%d/%Y %H:%M:%S.%f")
    ecf_start = ecf_start_obj.timestamp()

    ecfdf = pd.read_csv(ecf_text_filename, decimal=',', names=['t', 'U', 'I'],
                        sep='\t', skiprows=Nb_header_lines, encoding='ANSI')

    return ecfdf, ecf_start
    
def read_photodiode(ph_filename):
    with open(ph_filename) as file:
        ph_start_str = file.readline().split(sep='\t')[-1][:-1]
    date_obj = datetime.strptime(ph_start_str, "%Y-%m-%d %H:%M:%S")
    ph_start = date_obj.timestamp()

    phdf = pd.read_csv(ph_filename, sep='\t', decimal=',', skiprows=1, header=None)
    phdf.columns = ['t', 'photodiode']
    phdf.t = phdf.t.astype(float)
    return phdf, ph_start



def optimise_emsi_start(ecfdf, ecf_start, emsidf, emsi_start):
    roi_start = max(ecf_start, emsi_start) + 600
    roi_end = roi_start + 6600
    ecf_useful = ecfdf.loc[(ecfdf.t > roi_start-ecf_start) & (ecfdf.t < roi_end-ecf_start)]
    emsi_useful = emsidf.loc[(emsidf.t > roi_start-10-emsi_start) & (emsidf.t < roi_end+10-emsi_start)]
    emsi_useful.loc[:, 't'] = emsi_useful.t+emsi_start-roi_start
    ecf_useful.loc[:, 't'] = ecf_useful.t+ecf_start-roi_start

    def fit(shift: float) -> float:
        loss = np.sum(np.square(1000*emsi_useful.I - np.interp(emsi_useful.t, ecf_useful.t+shift, ecf_useful.I)))
        return loss
    shift = minimize(fit, 1).x
    print(shift)
    return shift

def alt_opt_emsi_start(ecfdf, ecf_start, emsidf, emsi_start):
    roi_start = max(ecf_start, emsi_start) + 600
    roi_end = roi_start + 6600
    ecf_useful = ecfdf.loc[(ecfdf.t > roi_start-ecf_start) & (ecfdf.t < roi_end-ecf_start)]
    emsi_useful = emsidf.loc[(emsidf.t > roi_start-10-emsi_start) & (emsidf.t < roi_end+10-emsi_start)]
    emsi_useful.loc[:, 't'] = emsi_useful.t+emsi_start-roi_start
    ecf_useful.loc[:, 't'] = ecf_useful.t+ecf_start-roi_start
    emsi_useful = emsi_useful.reset_index(drop=True)
    ecf_useful = ecf_useful.reset_index(drop=True)

    emsi_peak_indicies = find_peaks(np.diff(emsi_useful.I, n=2, prepend=0, append=0), height=0.01)[0]
    emsi_spikes = np.array(emsi_useful.loc[emsi_peak_indicies, 't'])
    ecf_peak_indicies = find_peaks(np.diff(ecf_useful.I, n=2, prepend=0, append=0), height=0.01)[0]
    ecf_spikes = np.array(ecf_useful.loc[ecf_peak_indicies, 't'])

    emsi_spikes = np.array(emsi_useful.t[find_peaks(emsi_useful.I, height = 0.035, prominence = 0.02)[0]])
    ecf_spikes = np.array(ecf_useful.t[find_peaks(ecf_useful.I, height = 0.035, prominence = 0.02)[0]])

    print(ecf_spikes)
    print(emsi_spikes)
    def fit(shift:float) -> float:
        loss = np.median(np.min(np.abs(np.subtract.outer(emsi_spikes, ecf_spikes+shift)), axis=1))
        return loss
    shift = minimize(fit, 1).x
    print(shift)
    return shift

def optimise_emsi_start_voltage(ecfdf, ecf_start, emsidf, emsi_start, basis_voltage, pert_strength):
    roi_start = max(ecf_start, emsi_start) + 600
    roi_end = roi_start + min(ecfdf.t.iloc[-1], emsidf.t.iloc[-1]) - 1200
    ecf_useful = ecfdf.loc[(ecfdf.t > roi_start-ecf_start) & (ecfdf.t < roi_end-ecf_start)].reset_index(drop=True)
    emsi_useful = emsidf.loc[(emsidf.t > roi_start-10-emsi_start) & (emsidf.t < roi_end+10-emsi_start)].reset_index(drop=True)
    emsi_useful.U = gaussian_filter1d(emsi_useful.U, 2)
    emsi_useful.loc[:, 't'] = emsi_useful.t+emsi_start-roi_start
    ecf_useful.loc[:, 't'] = ecf_useful.t+ecf_start-roi_start

    peak_indicies = find_peaks(ecf_useful.U*np.sign(pert_strength) - ecf_useful.t)[0]
    peak_times = np.array(ecf_useful.loc[peak_indicies, 't'])
    peak_times_ecf = peak_times[np.abs(ecf_useful.U[ecf_useful.t.isin(peak_times)]-(basis_voltage+pert_strength)) < 0.1]

    peak_indicies = find_peaks(emsi_useful.U*np.sign(pert_strength) - 5*emsi_useful.t)[0]
    peak_times = np.array(emsi_useful.loc[peak_indicies, 't'])
    peak_times_emsi = peak_times[np.abs(emsi_useful.U[emsi_useful.t.isin(peak_times)]-(basis_voltage+pert_strength)) < 0.1]
    if peak_times_emsi[0] < 0:
        peak_times_emsi = peak_times_emsi[1:]
    if peak_times_emsi.size > peak_times_ecf.size:
        peak_times_emsi = peak_times_emsi[:-1]
    shift = np.median(peak_times_emsi-peak_times_ecf)
    print(shift)
    return shift