from read_VA import read_VA
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def find_cycles(df, threshold_I_multiplier=0.9, sign_change = -1):
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
    unpurged_cycles = pd.DataFrame('start': crossing_times,
                           'duration': period_durations)
    
    # Remove false crossings (when period is too short)
    mean_period = np.mean(unpurged_cycles['duration'])
    cycles = unpurged_cycles[(unpurged_cycles['duration'] > 0.9*mean_period)]
    cycles['duration'] = np.diff(df['start']) #DEBUG Check df.assign
    cycles.drop(cycles.tail(1).index, inplace=True) # Drop last row
    
    return cycles

# =============================================================================
# plt.plot(df['t'], df['I'])
# plt.hlines(mean_current, 100, 5000, colors='r')
# plt.scatter(crossings['t'], crossings['I'], c='g', marker='x')
# plt.xlim(2400, 2500)
# plt.show() 
# =============================================================================

def pert_response(crossings, mean_period, pert, periods):
    '''
    Calculate the 
    '''
    indicies = np.searchsorted(crossings, np.array(pert))-1
    phase = (pert-crossings[indicies])/mean_period
    
    
    xd = np.array([periods[x:x+2] for x in indicies[:-1]])
    
    response = np.sum(xd, axis=1)-2*mean_period
    
    return phase[:-1], response

def phase_correction(df, crossings):
    spikes, _ = find_peaks(df['I'], height=0.1, distance=1000)
    time_array = np.array(df['t'])
    times = time_array[spikes]
    return np.average(times[:crossings.size]-crossings)








if __name__ == '__main__':






    df, pert = read_VA('T:\\Team\\Szewczyk\\Data\\20231103\\A00201_C01.txt',
                         margins=(300, 4100), p_height=0.1)
    periods = find_periods(df)
    
    phase, response = pert_response(crossings, np.average(periods['duration']), pert, periods['duration'])
    phase = (phase-phase_correction(df, crossings)/np.average(periods['duration']))%1
    plt.figure()
    plt.scatter(phase, response)
    # plt.title("INVALID FIT!!! DON'T USE")
    

    plt.figure()
    plt.scatter(crossings, periods['duration'], marker='+')
    plt.plot(crossings, periods['duration'])
    plt.vlines(pert, 40, 50, colors='r', linestyles='dashed')
    
    
    plt.ylim(41, 42)
    plt.title('Perturbation: +1V, 0.1s')
    plt.xlabel('Time [s]')
    plt.ylabel('Osc. period [s]')