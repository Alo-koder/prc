from read_data import read_data
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import numpy as np

df, pert = read_data('T:\\Team\\Szewczyk\\Data\\20231024\\raw\\A00102.dat',
                     margins=(1500, 2500))

peak_indicies, _ = find_peaks(df['EMSI'], distance=100, prominence=0.1)
peaks = df.iloc[peak_indicies]

plt.plot(df['t'], df['EMSI'])
plt.scatter(peaks['t'], peaks['EMSI'], marker='x', c='r')
plt.vlines(pert['t'], 15.7, 16.6, linestyle='dashed', colors='y')
plt.show()

peak_times = np.array(peaks['t'])
periods = (peak_times-np.roll(peak_times, 1))[1:]