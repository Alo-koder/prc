'''
Import the measurement metadata (such as basis voltage, pert. duration etc.) from a config file.

The DEVICE and DATE constants are meant to be changed by the user.
`filenames_dict` specifies file locations:
'data' is the folder from which the datafile will be read;
'notes' specifies where the plots will be saved
'''


import munch
import yaml

# CHANGE CONFIG HERE
DEVICE = 'tum'
DATE = '03-12'
# CHANGE CONFIG HERE

if DEVICE == 'laptop':
    filenames_dict = {
        'data'  : '/mnt/Szewczyk/Data/2024-'+DATE+'/',
        'notes' : '/home/alo/OneDrive/Notes/Lab/experiments/plots/'+DATE[1:]+'/'
    }

if DEVICE == 'laptop_local':
    filenames_dict = {
        'data'  : '/run/media/alo/0CD4823BD48226D4/Szewczyk/Data/2024-'+DATE+'/',
        'notes' : '/home/alo/OneDrive/Notes/Lab/experiments/plots/'+DATE[1:]+'/'
    }

elif DEVICE == 'tum':
    filenames_dict = {
        'data'  : 'T:/Team/Szewczyk/Data/2024-'+DATE+'/',
        'notes' : 'C:/Users/go68nim/OneDrive - University of Edinburgh/Notes/Lab/experiments/plots/'+DATE[1:]+'/'
    }

elif DEVICE == 'tum_local':
    filenames_dict = {
        'data'  : 'D:/Alek/Data/'+DATE[1:]+'-',
        'notes' : 'C:/Users/go68nim/OneDrive - University of Edinburgh/Notes/Lab/experiments/plots/'+DATE[1:]+'/'
    }

else:
    raise ValueError('Invalid device config.')

with open(filenames_dict['data'] + 'properties.yaml') as file:
    props_dict = yaml.safe_load(file)

defaults = {
    'bad_data': [],
    'max_period': 80,
    'interpolation': 'cubic',
    'period_measurement': 'peaks',
    'expected_period': 'polyfit',
}

props_dict = defaults | props_dict # this means: if a property isn't specified in the config file, use the default

filenames:munch.Munch = munch.munchify(filenames_dict)
props:munch.Munch = munch.munchify(props_dict)
