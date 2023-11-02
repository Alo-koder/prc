import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import numpy as np

def read_data(filename, p_height=0.5, margins=(100, 5000)):
    
    df = pd.read_csv(filename, sep='\t', header=5)
    
    
    # Renaming columns
    new_cols = ['t', 'U', 'I', 'Ill', 'EMSI']
    old_cols = df.columns
    col_dict = {x: y for x, y in zip(old_cols, new_cols)}
    df = df.rename(columns = col_dict)
    
    # Flipping signs
    df['U'] = -df['U']
    df['I'] = -df['I']
    
    # Applying margins
    df = df[(df['t'] > margins[0]) & (df['t'] < margins[1])]
    
    # Signal smoothening
    df['I'] = gaussian_filter(df['I'], 5)
    df['EMSI'] = gaussian_filter(df['EMSI'], 2)
    # df['U'] = gaussian_filter(df['U'], 1)
    
    # Finding voltage spikes
    spike_indicies, _ = find_peaks(np.sign(p_height) * df['U'],
                                   prominence=np.abs(p_height)*0.8)
    spikes = df.iloc[spike_indicies]
    
    return df, spikes

if __name__ == '__main__':
    df, spikes = read_data('T:\\Team\\Szewczyk\\Data\\20231024\\raw\\A00102.dat')
    
    from matplotlib import pyplot as plt
    
    # plt.xlim((4080, 4100))
    
    plt.plot(df['t'], df['U'])
    plt.scatter(spikes['t'], spikes['U'], marker='x', c='r')
    plt.show()
    
    