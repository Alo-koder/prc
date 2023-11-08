#from read_data import read_data
from read_VA import read_VA
#from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import numpy as np

def find_periods(df):
    
    mean_current = df.mean()['I']
    df['I relative'] = df['I']-mean_current
    crossings = df[(np.diff(np.sign(df['I relative']), append=-1) > 0)]
    periods = np.diff(crossings['t'])
    
    return periods, np.array(crossings['t'][:-1])

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
    
    # This is not valid!!! we need to wait until the line stabilises
    response = periods[indicies+1]-mean_period
    # This is not valid!
    return phase, response

if __name__ == '__main__':
    df, pert = read_VA('T:\\Team\\Szewczyk\\Data\\20231103\\A00202_C01.txt',
                         margins=(400, 4100), p_height=0.1)
    periods, crossings = find_periods(df)
    
    phase, response = pert_response(crossings, np.average(periods), pert, periods)
    plt.scatter(phase, response)
    plt.title("INVALID FIT!!! DON'T FORGET")
    plt.show()
    
    plt.scatter(crossings, periods, marker='+')
    plt.plot(crossings, periods)
    plt.vlines(pert, 40, 50, colors='r', linestyles='dashed')
    
    plt.xlim(1400, 1800)
    plt.ylim(40, 43)
    plt.title('Perturbation: +1V, 0.1s')
    plt.xlabel('Time [s]')
    plt.ylabel('Osc. period [s]')
    plt.show()
    
    plt.xlim(500, 700)
    plt.plot(df['t'], df['I'])
    plt.xlabel('Time [s]')
    plt.ylabel('Current [A]')
    plt.vlines(pert, 0.02, 0.14, colors='r')
    plt.vlines(crossings, 0.02, 0.14, colors='g', linestyles='dashdot')
    plt.show()