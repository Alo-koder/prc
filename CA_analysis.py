import eclabfiles as ecf
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
from scipy.signal import find_peaks


#params
filename    = 'T:\\Team\\Szewczyk\\Data\\20231128\\A00401_C01.mpr'
roi         = (2000, 99999)
p_height    = 0.3
#\params

def read_data(filename=filename, roi=roi, p_height=p_height):
    
    raw_data = ecf.to_df(filename)
    
    # Renaming columns
    col_dict = {'time': 't', 'Ewe': 'U', '<I>': 'I'}
    data = raw_data.rename(columns = col_dict)
    
    # Applying margins
    data = data[(data['t']>roi[0]) & (data['t']<roi[1])]
    
    # Signal smoothening
    data['I'] = gaussian_filter(data['I'], 10)

    spike_indicies, _ = find_peaks(np.sign(p_height) * data['U'],
                                   prominence=np.abs(p_height)*0.8, distance=1000)
    spikes = data.iloc[spike_indicies]
    
    return data.reset_index(), np.array(spikes['t'])

def find_periods(data):
    
    mean_current = data.mean()['I']
    data['I relative'] = data['I']-mean_current
    crossings = data[(np.diff(np.sign(data['I relative']), append=-1) > 0)]
    
    crossings = np.array(crossings['t'])
    periods = np.diff(crossings)
    
    return periods, np.array(crossings[:-1])


def period_fit(periods, crossings):
    smooth_periods = gaussian_filter(periods, 20)
    return smooth_periods

if __name__ == '__main__':
    data=read_data()
    periods, crossings = find_periods(data)
    plt.plot(data['t'], data['I'])
    plt.show()

    smooth_periods = period_fit(periods, crossings)
    plt.plot(crossings, periods)
    plt.plot(crossings, smooth_periods, c='r', linestyle='dashed')
    plt.show()











