from scipy.fft import fft, fftfreq
import pandas as pd
import numpy as np

def find_frequency(df, *args):
    df = df[30_000:55_000]
    N = df.shape[0]
    d = 0.1
    xf = fftfreq(N, d)[1:N//2]
    
    emsi = np.array(df['EMSI'])
    
    yf = np.abs(fft(emsi)[1:N//2])
    
    freq = xf[np.argmax(yf)]
    period = 1/freq
    print(f'Period = {period:.3f}')
    
    return xf, yf, period