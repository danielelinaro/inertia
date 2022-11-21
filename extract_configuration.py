
import os
import sys
import json
import pickle

EXPERIMENTS_DIR = 'experiments/neural_network'
defaults = {'group': 1}

if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print(f'usage: {sys.argv[0]} experiment_ID configuration_template')
        sys.exit(1)

    experiment_ID = sys.argv[1]
    config_template = sys.argv[2]

    experiment_dir = os.path.join(EXPERIMENTS_DIR, experiment_ID)
    if not os.path.isdir(experiment_dir):
        print(f'{experiment_dir}: no such directory.')
        sys.exit(2)

    params_file = os.path.join(experiment_dir, 'parameters.pkl')
    if not os.path.isfile(params_file):
        print(f'{params_file}: no such file.')
        sys.exit(3)

    if not os.path.isfile(config_template):
        print(f'{config_template}: no such file.')
        sys.exit(4)

    params = pickle.load(open(params_file, 'rb'))
    config = json.load(open(config_template, 'r'))
    for key in config:
        if key not in params:
            print(f'`{key}` not in saved configuration, using default value ({defaults[key]}).')
            config[key] = defaults[key]
        else:
            config[key] = params[key]

    json.dump(config, open('config_' + experiment_ID[:6] + '.json', 'w'), indent=4)

