import pandas as pd
import numpy as np
from scipy.signal import find_peaks, convolve
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from config import props


def find_perts(data:pd.DataFrame) -> np.ndarray:
    '''
    Find the perturbation times using voltage or light data.
    '''
    if props.pert_type == 'light':
        # peak_indicies = find_peaks(data.light*np.sign(props.pert_strength) - data.t)[0]
        # peak_times = np.array(data.loc[peak_indicies, 't'])
        peak_indicies = find_peaks(np.diff(data.I, n=2, prepend=0, append=0), height=0.0001)[0]
        peak_times = np.array(data.loc[peak_indicies, 't'])
        peak_times = peak_times[np.diff(peak_times, prepend=0) > 1]
        return peak_times
    elif props.pert_type == 'U':
        peak_indicies = find_peaks(data.U*np.sign(props.pert_strength) - data.t)[0]
        peak_times = np.array(data.loc[peak_indicies, 't'])
        peak_times = peak_times[np.abs(data.U[data.t.isin(peak_times)]-(props.voltage+props.pert_strength)) < 0.1]
        return peak_times
    else:
        raise ValueError(f'Invalid perturbation type: {props.pert_type}')



def data_cleaning(data:pd.DataFrame):
    '''
    Prepare the data for analysis by removing noise and (potentially)
    removing the perturbations from the current signal.
    '''
    data = data[data.t-data.t[0] > 300]
    for start, end in props.bad_data:
        data = data[((data.t<start) | (data.t>end))]
    data = data.reset_index(drop = True)
    data.I = convolve(data.I, [0.5, 0.5])[:-1]
    return data
    
def data_interpolation(data:pd.DataFrame, pert_times:np.ndarray):
    if props.interpolation == 'linear':
        for t in pert_times:
            data.loc[(data['t'] > t-0.1) & (data['t'] < t+props.pert_dt+0.1), 'I'] = np.nan
        data['I'] = np.interp(data['t'], data.loc[data['I'].notna() ,'t'], data.loc[data['I'].notna(), 'I'])

    elif props.interpolation == 'cubic':
        for t in pert_times:
            surrounding = data[((data.t>t-20) & (data.t<t-0.1)) | ((data.t>t+4) & (data.t<t+24))]
            fit = CubicSpline(surrounding.t, surrounding.I)
            affected = data[(data['t'] > t-0.1) & (data['t'] < t+4)]
            data.loc[(data['t'] > t-0.1) & (data['t'] < t+4), 'I'] = fit(affected.t)
    
    else:
        raise ValueError(f'Invalid interpolation type: {props.interpolation}')

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
    # det_points = globals()[f'_{props.find_cycles_method}'](data, pert_times)
    det_points = globals()[f'_find_cycles_{props.period_measurement}'](data, pert_times)
    period_durations = np.diff(det_points, append = np.nan)

    period_fit = np.polyfit(det_points[:-1], period_durations[:-1], 2)
    expected_duration = np.polyval(period_fit, det_points)

    perturbed_periods = np.searchsorted(det_points, np.array(pert_times))-1
    perturbed_periods = perturbed_periods[perturbed_periods>=0]

    cycles = pd.DataFrame({
                            'start'             : det_points,
                            'duration'          : period_durations,
                            'expected_duration' : expected_duration,
                            'had_pert'          : False,
                        })
    cycles.loc[perturbed_periods, 'had_pert'] = True

    # Purging bad cycles
    cycles.drop(cycles.tail(1).index, inplace=True) # Drop last row
    if (cycles.isna().any(axis=None) or (cycles['duration'] > props.max_period).any()):
            print(f'Warning! Some info might be wrong/missing')
            print(cycles[cycles.isna().any(axis=1)])
            print(cycles[cycles['duration'] > props.max_period])
    cycles = cycles[~cycles.isna().any(axis=1)]
    cycles = cycles[cycles['duration'] < props.max_period]


    # Recalculate expected duration
    period_fit = np.polyfit(cycles['start'], cycles['duration'], 2)
    cycles['expected_duration'] = np.polyval(period_fit, cycles['start'])

    cycles.reset_index(drop=True, inplace=True)
    return cycles

def _find_cycles_crossings(data:pd.DataFrame, pert_times:np.ndarray,
                           phase_det_direction:int = 1,
                           threshold_I_multiplier:float = 1.0):
    '''
    Cycle determination based on the current crossing a threshold value.
    The outcome can be affected by oscillation shape changes.
    '''

    # Calculate current relative to the threshold
    threshold_I = data.I.mean()*threshold_I_multiplier
    data['I_relative'] = data.I-threshold_I

    # Calculate crossings
    crossings = data[(np.diff(np.sign(data.I_relative), append=0) == 2*phase_det_direction)]
    return np.array(crossings.t)

def _find_cycles_peaks(data:pd.DataFrame, pert_times:np.ndarray):
    # Current characteristics:
    bottom_current, top_current = np.percentile(data[data.I.notna()].I, [1, 99])
    current_range = top_current - bottom_current

    # peak_indicies, _ = find_peaks(-data.I, height = -1.1*bottom_current, prominence=0.5*current_range)
    peak_indicies, _ = find_peaks(data.I, height = 0.7*top_current, prominence=0.7*current_range)
    return np.array(data.loc[peak_indicies, 't'])

def phase_correction(data, perts, cycles):
    '''
    Account for different phase determination method by offseting phase
    such that max current means phase = 0.

    Parameters
    ----------
    data : pd.DataFrame
        Experimental data
    perts_pos : pd.DataFrame
        Data on the perturbations
    cycles : pd.DataFrame
        Period data
    mean_period : float
        Mean cycle duration in seconds

    Returns
    -------
    perts : pd.DataFrame
        Updated perts dataframe with a new column: corrected_phase
    correction : float
        Average osc phase of current spikes (calculated with a relative method)
    '''
    spikes, _ = find_peaks(data['I'], height=0.03, distance=1000)
    spike_times = data['t'].iloc[spikes[10:-2]].reset_index(drop=True)
    in_which_period = np.searchsorted(cycles['start'], np.array(spike_times))-1

    cycles_useful = cycles.iloc[in_which_period].reset_index(drop=True)

    phase = (spike_times-cycles_useful['start'])/cycles_useful['expected_duration']

    if (phase>1.5).any():
        print('Warning! Bad spikes data.')
        print(phase[phase>1.5])
    spike_times = spike_times[(phase<1.5)]
    phase = phase[(phase<1.5)]

    phase_fit = np.polyfit(spike_times, phase, 5)
    correction = np.polyval(phase_fit, perts.time)


    print(f'{np.nanmedian(correction) = }')
    corrected_phase = (perts['phase']-correction)%1
    perts = perts.assign(corrected_phase = corrected_phase)

    return perts


def pert_response(data, cycles, pert_times):
    '''
    Create a dataframe with data about the perturbations.

    Parameters
    ----------
    cycles : pd.DataFrame
        A dataframe describing period information
            start -- t at which phase == 0
            duration -- T of this period
    pert_times : np.ndarray
    pert_direction : np.ndarray

    Returns
    -------
    perts : pd.DataFrame
        A dataframe describing information about each perturbation
            time -- start of the perturbation
            which_period -- index of the cycle in which pert occured
            phase -- osc phase at which pert occured relative to I crossing
            response -- phase response over current and next period
                as a fraction of a mean period
    '''

    which_period = np.searchsorted(cycles['start'], np.array(pert_times))-1
    
    period_fit = np.polyfit(cycles['start'], cycles['duration'], 5) #change back to 5!
    expected_period = np.polyval(period_fit, pert_times)
    # expected_period = np.average([cycles.duration[which_period-i] for i in range(1,5)], axis=0)
    # expected_period = np.array(cycles.duration[which_period-1])
    # smoothened_periods = gaussian_filter1d(cycles.duration, 15)
    # expected_period = np.interp(pert_times, cycles.start, smoothened_periods)

    phase = (pert_times-cycles['start'].iloc[which_period])/expected_period

    response = []
    duration = np.array(cycles['duration'])
    basis = -(duration[which_period-1]-expected_period)/expected_period
    for i in range(4):
        response.append(-(duration[which_period+i]-expected_period)/expected_period)

    perts = pd.DataFrame({'time'            : pert_times,
                        'which_period'      : which_period,
                        'phase'             : phase,
                        'basis'             : basis,
                        'response'          : np.sum(response[0:2], axis=0),
                        'response_0'        : response[0],
                        'response_1'        : response[1],
                        'response_2'        : response[2],
                        'response_3'        : response[3],
                        'expected_period'   : expected_period,
                        })

    if props.period_measurement == 'crossings':
        perts = phase_correction(data, perts, cycles)
    return perts