import munch
import yaml

DATE = '03-21'

filenames_dict = {
    'data' : 'T:/Team/Szewczyk/Data/2024-'+DATE,
    'notes' : 'C:/Users/go68nim/OneDrive - University of Edinburgh/Notes/Lab/experiments/plots/'+DATE[1:]
}

with open('T:/Team/Szewczyk/Data/2024-'+DATE+'/properties.yaml') as file:
    props_dict = yaml.safe_load(file)

filenames:munch.Munch = munch.munchify(filenames_dict)
props:munch.Munch = munch.munchify(props_dict)