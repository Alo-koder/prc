#from read_data import read_data
from read_VA import read_VA
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import numpy as np

def find_periods(df):
    
    mean_current = df.mean()['I']*0.9
    df['I relative'] = df['I']-mean_current
    crossings = df[(np.diff(np.sign(df['I relative']), append=1) < 0)]
    
    crossings = np.array(crossings['t'])
    periods = np.diff(crossings)
    
    return periods, np.array(crossings[:-1])

# =============================================================================
# plt.plot(df['t'], df['I'])
# plt.hlines(mean_current, 100, 5000, colors='r')
# plt.scatter(crossings['t'], crossings['I'], c='g', marker='x')
# plt.xlim(2400, 2500)
# plt.show() 
# =============================================================================

def pert_response(crossings, mean_period, pert, periods):
    indicies = np.searchsorted(crossings, np.array(pert))-1
    phase = (pert-crossings[indicies])/mean_period
    
    
    xd = np.array([periods[x:x+4] for x in indicies[:-1]])
    
    response1 = xd[:,0]-mean_period
    response2 = xd[:,1]-mean_period
    response3 = xd[:,2]-mean_period
    response4 = xd[:,3]-mean_period
    
    return phase[:-1], response1, response2, response3, response4

def phase_correction(df, crossings):
    spikes, _ = find_peaks(df['I'], height=0.1, distance=1000)
    time_array = np.array(df['t'])
    times = time_array[spikes]
    size = min(times.size, crossings.size)
    return np.average(times[:size]-crossings[:size])

if __name__ == '__main__':
    df, pert = read_VA('T:\\Team\\Szewczyk\\Data\\20231103\\A00202_C01.txt',
                         margins=(300, 4100), p_height=0.1)
    periods, crossings = find_periods(df)
    
    mean_period = np.average(periods[periods>30])

    phase, response1, response2, response3, response4 = pert_response(crossings, mean_period, pert, periods)
    phase = (phase-phase_correction(df, crossings)/mean_period)%1
    
    index_arr = np.argsort(phase)
    phase = np.array(phase)[index_arr]
    response1 = np.array(response1)[index_arr]
    response2 = np.array(response2)[index_arr]
    response3 = np.array(response3)[index_arr]
    response4 = np.array(response4)[index_arr]
    
    plt.figure()
    plt.plot(phase, response1, label='1st period')
    plt.plot(phase, response2, c='g', label='2nd period')
    plt.plot(phase, response3, c='r', label='3rd period')
    plt.plot(phase, response4, c='y', label='4th period')
    plt.legend()
    
    plt.figure()
    plt.scatter(crossings, periods, marker='+')
    plt.plot(crossings, periods)
    plt.vlines(pert, 40, 50, colors='r', linestyles='dashed')
    

    plt.ylim(41, 42)
    plt.title('Perturbation: +1V, 0.1s')
    plt.xlabel('Time [s]')
    plt.ylabel('Osc. period [s]')
    

    plt.figure()    
    plt.xlim(700, 1000)
    plt.hlines(df.mean()['I'], 1400, 1800, colors='y', linestyles='dashed')
    plt.plot(df['t'], df['I'])
    plt.xlabel('Time [s]')
    plt.ylabel('Current [A]')
    plt.scatter(pert, df[df['t'].isin(pert)]['I'], marker='x', c='r')
    plt.vlines(crossings, 0.02, 0.14, colors='g', linestyles='dashdot')
    plt.show()


    print('done')