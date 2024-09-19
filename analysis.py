'''
A collection of methods for analysing PRC determination experiments.

They functions are meant to be used within the <prc_analysis.ipynb> jupyter notebook.
'''


import pandas as pd
import numpy as np
from scipy.signal import find_peaks, convolve, sosfiltfilt, butter
from scipy.interpolate import CubicSpline, splrep, splev
from scipy.ndimage import gaussian_filter1d
from config import props
from matplotlib import pyplot as plt


def find_perts(data:pd.DataFrame) -> np.ndarray:
    '''
    Find the perturbations using voltage or light data.
    '''
    if props.pert_type == 'light':
        normal, perturbed = min(data.light*np.sign(props.pert_strength)), max(data.light*np.sign(props.pert_strength))
        middle = (normal + perturbed)/2
        centered = data.light-np.sign(props.pert_strength)*middle # signal that is negative at rest, positive at perturbation

        peak_times = data.t[np.diff(np.sign(centered), append=0) > 0]

        # The code below can be more accurate when perturbations cause a large current jump.
        # It looks for peaks of the second derivative of current.
        # Uncomment the 4 lines below if you want to try it out:

        # h = 0.001
        # peak_indicies = find_peaks(np.diff(data.I, n=2, prepend=0, append=0), height=h)[0]
        # peak_times = np.array(data.loc[peak_indicies, 't'])
        # peak_times = peak_times[np.diff(peak_times, prepend=0) > 2]

        print(f'Found {peak_times.size} perts')
        return peak_times
    
    elif props.pert_type == 'U':
        peak_indicies = find_peaks(data.U*np.sign(props.pert_strength) - data.t)[0]
        peak_times = np.array(data.loc[peak_indicies, 't'])
        peak_times = peak_times[np.abs(data.U[data.t.isin(peak_times)]-(props.voltage+props.pert_strength)) < 0.1]
        print(f'Found {peak_times.size} perts')
        return peak_times
    
    else:
        raise ValueError(f'Invalid perturbation type: {props.pert_type}')



def data_cleaning(data:pd.DataFrame) -> pd.DataFrame:
    '''
    Prepare the data for analysis by removing noise and (potentially)
    removing unusable parts of the current signal.
    '''
    data = data[data.t > 300]
    for start, end in props.bad_data:
        data = data[((data.t<start) | (data.t>end))]
    data = data.reset_index(drop = True)
    data['I_real'] = convolve(data.I, [0.5, 0.5])[:-1] # Removes 50Hz noise
    data.drop(columns='I')
    return data
    

def data_interpolation(data:pd.DataFrame, pert_times:np.ndarray) -> pd.DataFrame:
    '''
    Remove current outliers caused by perturbations from the current signal.

    The current signal around the perturbations is cut out
    and later interpolated from the surrounding.
    '''
    data['I'] = data.I_real.copy()
    if props.interpolation == 'linear':
        for t in pert_times:
            left_bound = t-0.2
            right_bound = t+props.pert_dt+0.2
            data.loc[(data.t > left_bound) & (data.t < right_bound), 'I'] = np.nan # nullify data between two bounds
        data['I'] = np.interp(data.t, data.loc[data.I_real.notna() ,'t'], data.loc[data.I.notna(), 'I_real'])

    elif props.interpolation == 'cubic': #TODO this can be improved with sine least square fitting
        for t in pert_times:
            surrounding = data[((data.t>t-20) & (data.t<t-0.2)) | ((data.t>t+3) & (data.t<t+24))]
            fit = CubicSpline(surrounding.t, surrounding.I)
            affected = data[(data['t'] > t-0.1) & (data['t'] < t+3)]
            data.loc[(data['t'] > t-0.1) & (data['t'] < t+3), 'I'] = fit(affected.t)
    
    elif props.interpolation == 'none':
        pass
    
    else:
        raise ValueError(f'Invalid interpolation type: {props.interpolation}')

    return data


def correct_emsi(data:pd.DataFrame) -> pd.DataFrame:
    '''
    Smoothen and normalise emsi data.

    Emsi is normalised to be between -0.5 and +0.5.
    '''
    true_emsi = np.array(data.emsi[::10])       # Emsi is currently recorded at 10Hz.
    sos = butter(10, 0.1, fs=10, output='sos')  # The 100Hz signal is linearly interpolated by the LabView program.
    filtered_emsi = sosfiltfilt(sos, true_emsi)
    emsi_long_term_fit = np.polyfit(data.t[::10], filtered_emsi, 2) # Removes some emsi drift; it's not perfect.
    tck = splrep(data.t[::10], filtered_emsi)
    high_res_emsi = splev(data.t, tck)
    emsi_corrected = high_res_emsi - np.polyval(emsi_long_term_fit, data.t)
    emsi_corrected = emsi_corrected/2/(np.percentile(emsi_corrected, 99) - np.percentile(emsi_corrected, 1))
    data['emsi_corrected'] = emsi_corrected
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
        Columns:
            start               : t at which phase = 0
            duration            : T of this cycle
            expected_duration   : predicted unperturbed T
            had_pert            : True if a perturbation occured
                                  within this period
    '''
    det_points = globals()[f'_find_cycles_{props.period_measurement}'](data)
    period_durations = np.diff(det_points, append = np.nan)
    amplitudes = np.array(data.I[data.t.isin(det_points)])
    amplitude_fit = np.polyfit(det_points, amplitudes, 4)
    expected_amplitude = np.polyval(amplitude_fit, det_points)

    period_fit = np.polyfit(det_points[:-1], period_durations[:-1], 2)
    expected_duration = np.polyval(period_fit, det_points)

    perturbed_periods = np.searchsorted(det_points, np.array(pert_times))-1
    perturbed_periods = perturbed_periods[perturbed_periods>=0]

    cycles = pd.DataFrame({
                            'start'             : det_points,
                            'duration'          : period_durations,
                            'expected_duration' : expected_duration,
                            'amplitude'         : amplitudes,
                            'expected_amplitude': expected_amplitude,
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
    cycles = cycles[cycles['duration'] > 20]


    # Recalculate expected duration
    period_fit = np.polyfit(cycles['start'], cycles['duration'], 2)
    cycles['expected_duration'] = np.polyval(period_fit, cycles['start'])

    cycles.reset_index(drop=True, inplace=True)
    return cycles

def _find_cycles_crossings(data:pd.DataFrame,
                           phase_det_direction:int = 1,
                           threshold_I_multiplier:float = 1.0):
    '''
    DEPRECATED Cycle determination based on the current crossing a threshold value.
    
    The outcome will be affected by oscillation shape changes.
    In general, not very accurate. Left here for consistency but shouldn't be used.
    '''
    # Calculate current relative to the threshold
    threshold_I = data.I.mean()*threshold_I_multiplier
    data['I_relative'] = data.I-threshold_I

    # Calculate crossings
    crossings = data[(np.diff(np.sign(data.I_relative), append=0) == 2*phase_det_direction)]
    return np.array(crossings.t)

def _find_cycles_peaks(data:pd.DataFrame):
    """
    Cycle determination based on current peaks.

    Currently the most accurate method. Should be used by default.
    """
    # Current characteristics:
    bottom_current, top_current = np.percentile(data[data.I.notna()].I, [1, 99])
    current_range = top_current - bottom_current

    peak_indicies, _ = find_peaks(data.I, height = 0.7*top_current, prominence=0.5*current_range)
    return np.array(data.loc[peak_indicies[1:-1], 't'])

def _find_cycles_emsi(data:pd.DataFrame):
    '''
    Cycle determination based on emsi minima.

    pro:  Doesn't rely on current fitting.
    con:  Slightly less accurate than current peaks.
    '''
    troughs, _ = find_peaks(-data.emsi_corrected)
    return np.array(data.t)[troughs[1:-1]]

def phase_correction_emsi_minimum(data, perts, cycles):
    '''
    Assign a new column: "corrected phase" to the `perts` dataframe.

    "corrected_phase" is the phase shifted such that on average
    the emsi minimum has phase = 0.
    '''
    troughs = _find_cycles_emsi(data)[5:]
    in_which_period = np.searchsorted(cycles['start'], np.array(troughs))-1
    cycles_useful = cycles.iloc[in_which_period].reset_index(drop=True) # cycles that were perturbed
    phase = (troughs-cycles_useful['start'])/cycles_useful['expected_duration']


    correction = np.nanmedian(phase)
    print(f'{np.nanmedian(phase) = }')
    corrected_phase = (perts['phase']-correction)%1
    perts = perts.assign(corrected_phase = corrected_phase)

    return perts

def phase_correction_current_maximum(data, perts, cycles):
    '''
    DEPRECATED Account for different phase determination method by offseting phase
    such that max current means phase = 0.

    Not used anymore -- phase is shifted either to emsi minimum <phase_correction_emsi_minimum>
    or left at phase==0 <=> current peak, as determined.

    Parameters
    ----------
    data    : pd.DataFrame
        Experimental data
    perts   : pd.DataFrame
        Info on all perturbations
    cycles  : pd.DataFrame
        Info on all periods

    Returns
    -------
    perts : pd.DataFrame
        Updated perts dataframe with a new column: corrected_phase
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
        A dataframe describing information about each perturbation.
        Columns:
            time            : start of the perturbation
            which_period    : index of the cycle in which pert occured
            phase           : osc phase at which pert occured relative to
                              the period determination point (usually current maximum)
            response        : phase response over current and next period
                              as a fraction of a mean period
    '''

    which_period = np.searchsorted(cycles['start'], np.array(pert_times))-1
    
    if props.expected_period == 'polyfit':
        period_fit = np.polyfit(cycles['start'], cycles['duration'], 5) #change back to 5!
        expected_period = np.polyval(period_fit, pert_times)
    elif props.expected_period == 'mean':
        expected_period = np.average([cycles.duration[which_period-i] for i in range(1,3)], axis=0)
    elif props.expected_period == 'previous':
        expected_period = np.array(cycles.duration[which_period-1])
    elif props.expected_period == 'gauss':
        smoothened_periods = gaussian_filter1d(cycles.duration, 15)
        expected_period = np.interp(pert_times, cycles.start, smoothened_periods)
    else:
        raise ValueError(f'Unknown expected period determination method: {props.expected_period}')

    phase = (pert_times-np.array(cycles['start'].iloc[which_period]))/expected_period

    phase_response = []
    duration = np.array(cycles['duration'])
    basis = -(duration[which_period-1]-expected_period)/expected_period
    for i in range(4):
        phase_response.append(-(duration[which_period+i]-expected_period)/expected_period)

    amplitude_response = []
    expected_amplitude = np.interp(pert_times, cycles.start, cycles.expected_amplitude)
    amplitudes = np.array(cycles.amplitude)
    basis_amplitude = (amplitudes[which_period]-expected_amplitude)/expected_amplitude
    for i in range(1,4):
        amplitude_response.append((amplitudes[which_period+i]-expected_amplitude)/expected_amplitude)

    perts = pd.DataFrame({'time'            : pert_times,
                        'which_period'      : which_period,
                        'phase'             : phase,
                        'basis'             : basis,
                        'response'          : np.sum(phase_response[0:2], axis=0),
                        'response_1'        : phase_response[0],
                        'response_2'        : phase_response[1],
                        'response_3'        : phase_response[2],
                        'response_4'        : phase_response[3],
                        'expected_period'   : expected_period,
                        'basis_amplitude'   : basis_amplitude,
                        'amp_response_1'    : amplitude_response[0],
                        'amp_response_2'    : amplitude_response[1],
                        'amp_response_3'    : amplitude_response[2],
                        })

    if props.period_measurement == 'crossings':
        perts = phase_correction_current_maximum(data, perts, cycles)
    return perts