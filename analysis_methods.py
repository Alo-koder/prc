import pandas as pd
import numpy as np
from scipy.signal import find_peaks, convolve
from config import props

def data_cleaning(data:pd.DataFrame, pert_times:np.ndarray):
    '''
    Prepare the data for analysis by removing noise and (potentially)
    removing the perturbations from the current signal.
    '''

    for start, end in props.bad_data:
        data = data.drop([(data.t>start) & (data.t<end)])
    data = data.reset_index(drop = True)
    
    data.I = convolve(data.I, [0.5, 0.5])


    return data




def find_cycles(data:pd.DataFrame, pert_times:np.ndarray):
    '''
    An abstract method for finding cycles. 
    
    Parameters
    ----------
    data        : pd.DataFrame
        All relevant experimental data
    pert_times  : np.ndarray
        Times of disturbances in the signal. Some implementations
        will ignore the perturbation current.
    props  : dict
        Properties of the experiment, as listed in the config.yaml file.
        Implementation is chosen based on them.        

    Returns
    -------
    cycles      : pd.DataFrame
        A dataframe describing period information.
        Index:
            start               : t at which phase = 0
        Columns:
            duration            : T of this cycle
            expected_duration   : predicted unperturbed T
            had_pert            : True if a perturbation occured
                                  within this period
    '''
    det_points = globals()[f'_{props.find_cycles_method}'](data, pert_times)
    period_durations = np.diff(det_points, append = np.nan)

    period_fit = np.polyfit(det_points[:-1], period_durations[:-1], 2)
    expected_duration = np.polyval(period_fit, det_points)

    perturbed_periods = np.searchsorted(det_points, np.array(pert_times))-1

    cycles = pd.DataFrame({
                            'start'             : det_points,
                            'duration'          : period_durations,
                            'expected_duration' : expected_duration,
                            'had_pert'          : False,
                        })
    cycles.loc[perturbed_periods, 'had_pert'] = True

    # Purging bad cycles
    cycles.drop(cycles.tail(1).index, inplace=True) # Drop last row
    if (cycles.isna().any(axis=None) or (cycles['duration'] > props.max_duration).any()):
            print(f'Warning! Some info might be wrong/missing')
            print(cycles[cycles.isna().any(axis=1)])
            print(cycles[cycles['duration'] > props.max_duration])
    cycles = cycles[~cycles.isna().any(axis=1)]
    cycles = cycles[cycles['duration'] < props.max_duration]


    # Recalculate expected duration
    period_fit = np.polyfit(cycles['start'], cycles['duration'], 2)
    cycles['expected_duration'] = np.polyval(period_fit, cycles['start'])

    cycles.reset_index(drop=True, inplace=True)
    return cycles

def _find_cycles_crossings(data:pd.DataFrame, pert_times:np.ndarray,
                           phase_det_direction:int = -1,
                           threshold_I_multiplier:float = 1.0):
    '''
    Cycle determination based on the current crossing a threshold value.
    The outcome can be affected by oscillation shape changes.
    '''

    # Calculate current relative to the threshold
    threshold_I = data.I.mean()*threshold_I_multiplier
    data['I_relative'] = data.I-threshold_I

    # Calculate crossings and create 'cycles' dataframe
    crossings = data[(np.diff(np.sign(data.I_relative), append=0) == 2*phase_det_direction)]

    