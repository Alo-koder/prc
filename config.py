import munch
import yaml

DEVICE = 'tum_local'
DATE = '03-21'

if DEVICE == 'laptop':
    filenames_dict = {
        'data' : '/mnt/Szewczyk/Data/2024-'+DATE+'/',
        'notes' : '~/Onedrive/Notes/Lab/experiments/plots/'+DATE[1:]
    }

elif DEVICE == 'tum':
    filenames_dict = {
        'data' : 'T:/Team/Szewczyk/Data/2024-'+DATE+'/',
        'notes' : 'C:/Users/go68nim/OneDrive - University of Edinburgh/Notes/Lab/experiments/plots/'+DATE[1:]
    }

elif DEVICE == 'tum_local':
    filenames_dict = {
        'data' : 'D:/Alek/Data/'+DATE[1:]+'-',
        'notes' : 'C:/Users/go68nim/OneDrive - University of Edinburgh/Notes/Lab/experiments/plots/'+DATE[1:]
    }

else:
    raise ValueError('Invalid device config.')

with open(filenames_dict['data'] + 'properties.yaml') as file:
    props_dict = yaml.safe_load(file)

filenames:munch.Munch = munch.munchify(filenames_dict)
props:munch.Munch = munch.munchify(props_dict)