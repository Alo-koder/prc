import pandas as pd
import eclabfiles as ecf
import numpy as np
from datetime import datetime

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