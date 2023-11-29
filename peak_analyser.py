from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from freq_finder import find_frequency
from read_data import read_data

# =============================================================================
# Here go config params

filename = 'T:\\Team\\Szewczyk\\Data\\20231128\\A00400.dat'
lower_cutoff = 1800
upper_cutoff = 2080
shift = 1826

# End of config
# =============================================================================


df, spikes = read_data(filename)




roi = df[(df['t'] > lower_cutoff) & (df['t'] < upper_cutoff)]



shifted = df.copy()
shifted = df.shift(periods = shift)
shifted['t'] += shift/10
roi_shifted = shifted[(shifted['t'] > lower_cutoff) & (shifted['t'] < upper_cutoff)]

plt.plot(roi['t'], roi['EMSI'])
plt.plot(roi_shifted['t'], roi_shifted['EMSI'])
plt.vlines((1825, 1825+shift/10), 15.9, 16.7, linestyles='dashed', colors='r')
plt.show()

diff = roi_shifted['EMSI'] - roi['EMSI']

# =============================================================================
# plt.ylim(-0.3, 0.3)
# plt.plot(roi['t'], diff*3)
# plt.plot(roi['t'], roi['U']+3.8)
# # plt.plot(roi['t']+shift/10, roi['U']+3.8)
# 
xf, yf, _ = find_frequency(df)
# # plt.xlim(0, 0.1)
# # plt.plot(xf, yf)
# =============================================================================

lower_cutoff = 1822
upper_cutoff = 1900

pert = df[(df['t'] > lower_cutoff) & (df['t'] < upper_cutoff)]
plt.plot(roi['I'], roi['EMSI'])
plt.plot(pert['I'], pert['EMSI'])



plt.show()

plt.plot(df['t'], df['U'])
plt.show()
