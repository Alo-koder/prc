import pandas as pd
import eclabfiles as ecf
import numpy as np
from datetime import datetime
from scipy.optimize import minimize

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