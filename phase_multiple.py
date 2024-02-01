#from read_data import read_data
from read_VA import read_VA
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import numpy as np

def find_periods(df, crossing_height):
    
    mean_current = df.mean()['I']*crossing_height
    df['I relative'] = df['I']-mean_current
    crossings = df[(np.diff(np.sign(df['I relative']), append=-1) < 0)]
    
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
    
    
    xd = np.array([periods[x:x+1] for x in indicies[:-1]]) #1.period only
    
    response = np.sum(xd, axis=1)-1*mean_period

    # response = periods[indicies]-mean_period
    
    return phase[:-1], response

def phase_correction(df, crossings):
    spikes, _ = find_peaks(df['I'], height=0.1, distance=1000)
    time_array = np.array(df['t'])
    times = time_array[spikes]
    return np.average(times[:53]-crossings[:53])


if __name__ == '__main__':
    df, pert = read_VA('~/munich/data/A00201_C01.txt',
                         margins=(300, 4100), p_height=0.1)
    phase_cors = []
    responses1 = []
    responses2 = []
    x=[]
    for crossing_height in np.linspace(0.9, 0.94, 41):


        periods, crossings = find_periods(df, crossing_height)
        phase, response = pert_response(crossings, np.average(periods), pert, periods)
        correction = phase_correction(df, crossings)/np.average(periods)
        phase_cor = (phase-correction)%1
        if crossing_height == 1:
            x.append(phase_cor[2])
        phase_cors.append((phase-correction)%1)
        responses1.append(response[:53])


    for crossing_height in np.linspace(1.05, 1.1, 51):


        periods, crossings = find_periods(df, crossing_height)
        phase, response = pert_response(crossings, np.average(periods), pert, periods)
        correction = phase_correction(df, crossings)/np.average(periods)
        phase_cor = (phase-correction)%1
        if crossing_height == 1:
            x.append(phase_cor[2])
        phase_cors.append((phase-correction)%1)
        responses2.append(response[:53])
    
    
    plt.figure()
    plt.scatter(np.array(phase_cors[:41]).flatten(), np.array(responses1).flatten(), s=1, c='r')
    plt.scatter(np.array(phase_cors[41:]).flatten(), np.array(responses2).flatten(), s=1, c='b')
    plt.title("Phase Response Curve")
    plt.xlabel('Phase of the perturbation')
    plt.ylabel('Period elongation [s]')
# =============================================================================
#     plt.scatter(crossings, periods, marker='+')
#     plt.plot(crossings, periods)
#     plt.vlines(pert, 40, 50, colors='r', linestyles='dashed')
#     
#     plt.xlim(2600, 2900)
#     plt.ylim(41, 42)
#     plt.title('Perturbation: +1V, 0.1s')
#     plt.xlabel('Time [s]')
#     plt.ylabel('Osc. period [s]')
#     plt.show()
# =============================================================================
    
    plt.figure()
    #plt.xlim(1000, 2000)
    plt.hlines(df.mean()['I'], 1400, 1800, colors='y', linestyles='dashed')
    plt.plot(df['t'], df['I'])
    plt.xlabel('Time [s]')
    plt.ylabel('Current [A]')
    plt.scatter(pert, df[df['t'].isin(pert)]['I'], marker='x', c='r')
    #plt.scatter(df.iloc[spikes]['t'], df.iloc[spikes]['I'], marker='x', c='r')
    plt.vlines(crossings, 0.02, 0.14, colors='g', linestyles='dashdot')


    plt.figure()
    plt.plot(df['t'], df['U'])
    plt.vlines(pert, 3.95, 4.15, colors='g', linestyles='dashdot')
    plt.show()


    print('done')
    