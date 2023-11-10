#from read_data import read_data
from read_VA import read_VA
#from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import numpy as np

def find_periods(df):
    
    mean_current = df.mean()['I']
    df['I relative'] = df['I']-mean_current
    crossings = df[(np.diff(np.sign(df['I relative']), append=-1) > 0)]
    
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
    
    response = np.sum(xd, axis=1)-4*mean_period
    
    return phase[:-1], response

if __name__ == '__main__':
    df, pert = read_VA('T:\\Team\\Szewczyk\\Data\\20231103\\A00202_C01.txt',
                         margins=(300, 4100), p_height=0.1)
    periods, crossings = find_periods(df)
    
    phase, response = pert_response(crossings, np.average(periods), pert, periods)
    plt.scatter(phase, response)
    plt.title("INVALID FIT!!! DON'T USE")
    plt.show()
    
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
    
    plt.xlim(700, 1000)
    plt.hlines(df.mean()['I'], 1400, 1800, colors='y', linestyles='dashed')
    plt.plot(df['t'], df['I'])
    plt.xlabel('Time [s]')
    plt.ylabel('Current [A]')
    plt.scatter(pert, df[df['t'].isin(pert)]['I'], marker='x', c='r')
    plt.vlines(crossings, 0.02, 0.14, colors='g', linestyles='dashdot')
    plt.show()
