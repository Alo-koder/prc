#from read_data import read_data
from read_VA import read_VA
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# def find_periods(df, threshold_I_multiplier=0.9):
    
#     threshold_I = df.mean()['I']*threshold_I_multiplier
#     df['I relative'] = df['I']-threshold_I
#     crossings = df[(np.diff(np.sign(df['I relative']), append=1) < 0)]
    
#     crossings = np.array(crossings['t'])
#     periods = np.diff(crossings)
    
#     return periods, np.array(crossings[:-1])

def find_cycles(df, threshold_I_multiplier=0.9, sign_change = 1):
    '''
    Divide the signal into cycles by cutting when the current
    crosses a specific value.

    Parameters
    ----------
    df : pd.DataFrame
        experimental data
    threshold_I_multiplier : float
        point where I == threshold_I is defined
        to have phase = 0; defined as fraction of mean current
    sign_change : int
        1 when I raises at phase==0, otherwise -1

    
    Returns
    -------
    cycles : pd.DataFrame
        A dataframe describing period information
            start -- t at which phase == 0
            duration -- T of this period
    '''
    
    # Calculate current relative to the threshold
    threshold_I = df.mean()['I']*threshold_I_multiplier
    df['I relative'] = df['I']-threshold_I

    # Calculate crossings and create 'cycles' dataframe
    crossings = df[(np.diff(np.sign(df['I relative']), append=0) == 2*sign_change )]
    crossing_times = np.array(crossings['t'])
    period_durations = np.diff(crossing_times)


    #DEBUG This may throw an error! Last entry in durations should be NaN
    unpurged_cycles = pd.DataFrame({'start': crossing_times,
                           'duration': np.append(period_durations, np.nan)})
    
    # Remove false crossings (when period is too short)
    mean_period = np.mean(unpurged_cycles['duration'])
    cycles = unpurged_cycles[(unpurged_cycles['duration'] > 0.9*mean_period)]
    cycles = cycles.assign(duration= np.diff(cycles['start'], append=np.nan)) #DEBUG Check df.assign
    cycles.drop(cycles.tail(1).index, inplace=True) # Drop last row
    cycles.reset_index(inplace=True)
    
    return cycles, threshold_I

# =============================================================================
# plt.plot(df['t'], df['I'])
# plt.hlines(mean_current, 100, 5000, colors='r')
# plt.scatter(crossings['t'], crossings['I'], c='g', marker='x')
# plt.xlim(2400, 2500)
# plt.show() 
# =============================================================================

# def pert_response(crossings, mean_period, pert, periods):
#     indicies = np.searchsorted(crossings, np.array(pert))-1
#     phase = (pert-crossings[indicies])/mean_period
    
    
#     xd = np.array([periods[x:x+2] for x in indicies[:-1]])
    
#     response = np.sum(xd, axis=1)-2*mean_period
    
#     return phase[:-1], response

def pert_response(cycles, pert_times):
    '''
    Create a dataframe with data about the perturbations.

    Parameters
    ----------
    cycles : pd.DataFrame
        A dataframe describing period information
            start -- t at which phase == 0
            duration -- T of this period
    pert_times : np.ndarray
        An array of values of time when perturbations happened

    Returns
    -------
    perts : pd.DataFrame
        A dataframe describing information about each perturbation
            time -- start of the perturbation
            in_which_period -- index of the cycle in which pert occured
            phase -- osc phase at which pert occured relative to I crossing
            # corrected_phase -- osc phase relative to current maximum
            response -- phase response over current and next period
                as a fraction of a mean period
    '''
    mean_period = np.mean(cycles['duration'])
    in_which_period = np.searchsorted(cycles['start'], np.array(pert_times))-1
    phase = (pert_times-cycles['start'].iloc[in_which_period])/mean_period
    
    
    affected_periods_durations = np.array([cycles['duration'].iloc[x:x+2] for x in in_which_period])
    response = np.sum(affected_periods_durations, axis=1)-2*mean_period
    
    perts = pd.DataFrame({'time': pert_times,
                                  'in_which_period': in_which_period,
                                  'phase': phase,
                                  'response': response})

    # return phase[:-1], response
    return perts

# def phase_correction(df, crossings):
#     spikes, _ = find_peaks(df['I'], height=0.1, distance=1000)
#     time_array = np.array(df['t'])
#     times = time_array[spikes]
#     return np.average(times[:crossings.size]-crossings)


def phase_correction(df, perts, cycles, mean_period):
    '''
    Account for different phase determination method by offseting phase
    such that max current means phase = 0.

    Parameters
    ----------
    df : pd.DataFrame
        Experimental data
    perts : pd.DataFrame
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
    spikes, _ = find_peaks(df['I'], height=0.1, distance=1000)
    time_array = np.array(df['t'])
    spike_times = time_array[spikes]
    size = min(spike_times.size, cycles.shape[0])
    correction = np.average(spike_times[:size]-cycles['start'].iloc[:size])%mean_period/mean_period
    corrected_phase = (perts['phase']-correction)%1
    perts.drop(perts.tail(perts.shape[0]-size).index, inplace=True)
    perts = perts.assign(corrected_phase = corrected_phase)

    return perts, correction

if __name__ == '__main__':

    # Import and prepare data
    df, pert_times = read_VA('T:\\Team\\Szewczyk\\Data\\20231103\\A00201_C01.txt',
                         margins=(300, 4100), p_height=0.1)
    cycles, threshold_I = find_cycles(df)
    mean_period = np.mean(cycles['duration'])

    # Calculate perturbation response
    perts = pert_response(cycles, pert_times)
    perts, correction = phase_correction(df, perts, cycles, mean_period)
    

    # Plot PRC
    plt.figure()
    plt.title("PRC -- INVALID FIT!!! DON'T USE")
    plt.xlabel('Phase rel. to current spike (fractional)')
    plt.ylabel('Phase response [s]')
    plt.scatter(perts['corrected_phase'], perts['response'])
    

    # Plot periods vs time
    plt.figure()
    plt.title('Period length versus time')
    plt.scatter(cycles['start'], cycles['duration'], marker='+')
    plt.plot(cycles['start'], cycles['duration'])
    for x in pert_times:
        plt.axvline(x, c='r', ls='--')
    plt.title('Perturbation: +1V, 0.1s')
    plt.xlabel('Time [s]')
    plt.ylabel('Osc. period [s]')
    

    # Plot current vs time
    plt.figure()
    plt.title('Current versus time')
    plt.axhline(threshold_I, c='y', ls='dashed')
    plt.plot(df['t'], df['I'])
    plt.xlabel('Time [s]')
    plt.ylabel('Current [A]')
    plt.scatter(pert_times, df[df['t'].isin(pert_times)]['I'], marker='x', c='r')
    for x in cycles['start']:
        plt.axvline(x, c='g', ls='-.')
    plt.show()


    print('done!')