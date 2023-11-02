from scipy.fft import fft, fftfreq
import pandas as pd
import numpy as np
from read_data import read_data
from matplotlib import pyplot as plt

def find_frequency(df):
    N = df.shape[0]
    d = 0.1
    xf = fftfreq(N, d)[1:N//2]
    
    emsi = np.array(df['EMSI'])
    
    yf = np.abs(fft(emsi)[1:N//2])
    
    freq = xf[np.argmax(yf)]
    period = 1/freq
    print(f'Period = {period:.3f}')
    
    return xf, yf, period

if __name__ == "__main__":
    df1, _ = read_data('T:\\Team\\Szewczyk\\Data\\20231024\\raw\\A00102.dat',
                   margins = (100, 2500))
    df2, _ = read_data('T:\\Team\\Szewczyk\\Data\\20231024\\raw\\A00102.dat',
                   margins = (2500, 4900))
    
    ft1 = find_frequency(df1)
    ft2 = find_frequency(df2)
    
    plt.plot(ft1[0], ft1[1])
    plt.plot(ft2[0], ft2[1], c='r')
    plt.xlim(0, 0.1)
    plt.show()
    