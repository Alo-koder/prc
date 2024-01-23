import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import numpy as np

def read_data(filename, p_height=0.5, margins=(1800, 1900)):
    
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
    df['I'] = gaussian_filter(df['I'], 10)*100_000
    df['EMSI'] = gaussian_filter(df['EMSI'], 2)
    # df['U'] = gaussian_filter(df['U'], 1)
    
    # Finding voltage spikes
    spike_indicies, _ = find_peaks(np.sign(p_height) * df['U'],
                                   prominence=np.abs(p_height)*0.8, distance=1000)
    spikes = df.iloc[spike_indicies]
    
    return df, np.array(spikes['t'])

if __name__ == '__main__':
    df, spikes = read_data('~/munich/data/24X/A00102.dat')
    
    from matplotlib import pyplot as plt
    
    # plt.xlim((4080, 4100))
    
    # fig, ax1 = plt.subplots()

    # ax1.set_xlabel('t [s]')
    
    # plt.plot(df['t'], df['U'])
    # plt.plot(df['t'], df['EMSI'])
    # #plt.scatter(spikes['t'], spikes['U'], marker='x', c='r')
    # plt.show()
    
    # fig, ax1 = plt.subplots(figsize=(5,3))

    # plt.xlim(2800, 3000)

    # color = 'tab:red'
    # ax1.set_xlabel('time [s]')
    # ax1.set_ylabel('current', color=color)
    # ax1.plot(df['t'], df['I'], color=color)
    # ax1.tick_params(axis='y', labelcolor=color)

    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # color = 'tab:blue'
    # ax2.set_ylabel('EMSI', color=color)  # we already handled the x-label with ax1
    # ax2.plot(df['t'], df['EMSI'], color=color)
    # ax2.tick_params(axis='y', labelcolor=color)

    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.show()
    
    print(spikes)
    fig, ax = plt.subplots()
    plt.plot(df['I'], df['EMSI'])

    ax.plot([0, 1], [0, 1], 'k:', transform=ax.transAxes)
    ax.plot([0, 1], [1, 0], 'k:', transform=ax.transAxes)
    ax.plot([0, 1], [0.5, 0.5], 'k:', transform=ax.transAxes)
    ax.plot([0.5, 0.5], [0, 1], 'k:', transform=ax.transAxes)

    plt.text(0.52, 0.75, r'$\phi = 0$', transform=ax.transAxes)
    plt.text(0.2, 0.52, r'$\phi = 0.25$', transform=ax.transAxes)
    plt.text(0.52, 0.25, r'$\phi = 0.5$', transform=ax.transAxes)
    plt.text(0.7, 0.52, r'$\phi = 0.75$', transform=ax.transAxes)

    plt.title('2D Phase Determination')
    plt.xlabel(r'Current [$\mu$A]')
    plt.ylabel('EMSI')
    plt.show()