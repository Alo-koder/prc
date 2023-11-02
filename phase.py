from read_data import read_data
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import numpy as np

def find_periods(df):
    
    mean_current = df.mean()['I']
    df['I relative'] = df['I']-mean_current
    crossings = df[(np.diff(np.sign(df['I relative']), append=-1) > 0)]
    periods = np.diff(crossings['t'])
    
    return periods, crossings['t'][:-1]

# =============================================================================
# plt.plot(df['t'], df['I'])
# plt.hlines(mean_current, 100, 5000, colors='r')
# plt.scatter(crossings['t'], crossings['I'], c='g', marker='x')
# plt.xlim(2400, 2500)
# plt.show() 
# =============================================================================

if __name__ == '__main__':
    df, pert = read_data('T:\\Team\\Szewczyk\\Data\\20231024\\raw\\A00102.dat',
                         margins=(2550, 3700), p_height=1)
    periods, crossings = find_periods(df)
    
    plt.scatter(crossings, periods, marker='+')
    plt.plot(crossings, periods)
    plt.vlines(pert['t'], 40, 50, colors='r', linestyles='dashed')
    
    plt.ylim(43, 47)
    plt.title('Perturbation: +1V, 0.1s')
    plt.xlabel('Time [s]')
    plt.ylabel('Osc. period [s]')
    plt.show()
    
    plt.plot(df['t'], df['I'])
    plt.xlabel('Time [s]')
    plt.ylabel('Current [A]')
    plt.vlines(pert['t'], 3e-6, 10e-6, colors='r', linestyles='dashed')
    plt.show()