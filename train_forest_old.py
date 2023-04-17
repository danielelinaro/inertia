import os
import sys
import re
import glob
import json
import argparse as arg
from time import strftime, localtime
import pickle

import numpy as np
from numpy.random import RandomState, SeedSequence, MT19937
import matplotlib.pyplot as plt
from comet_ml import Experiment

from sklearn.ensemble import RandomForestRegressor

from deep_utils import *

import colorama as cm
print_error   = lambda msg: print(f'{cm.Fore.RED}'    + msg + f'{cm.Style.RESET_ALL}')
print_warning = lambda msg: print(f'{cm.Fore.YELLOW}' + msg + f'{cm.Style.RESET_ALL}')
print_msg     = lambda msg: print(f'{cm.Fore.GREEN}'  + msg + f'{cm.Style.RESET_ALL}')


if __name__ == '__main__':

    progname = os.path.basename(sys.argv[0])
    
    parser = arg.ArgumentParser(description = 'Train a network to estimate inertia',
                                formatter_class = arg.ArgumentDefaultsHelpFormatter, \
                                prog = progname)
    parser.add_argument('config_file', type=str, action='store', help='configuration file')
    parser.add_argument('-o', '--output-dir',  default='experiments',  type=str, help='output directory')
    parser.add_argument('--no-comet', action='store_true', help='do not use CometML to log the experiment')
    args = parser.parse_args(args=sys.argv[1:])

    config_file = args.config_file
    if not os.path.isfile(config_file):
        print_error('{}: {}: no such file.'.format(progname, config_file))
        sys.exit(1)
    config = json.load(open(config_file, 'r'))

    with open('/dev/random', 'rb') as fid:
        seed = int.from_bytes(fid.read(4), 'little')
    rnd_state = RandomState(MT19937(SeedSequence(seed)))
    print_msg('Seed: {}'.format(seed))

    log_to_comet = not args.no_comet

    if log_to_comet:
        ### create a CometML experiment
        experiment = Experiment(
            api_key = os.environ['COMET_API_KEY'],
            project_name = 'inertia',
            workspace = 'danielelinaro'
        )
        experiment_key = experiment.get_key()
    else:
        experiment = None
        experiment_key = strftime('%Y%m%d-%H%M%S', localtime())

    ### generator IDs
    generator_IDs = config['generator_IDs']
    N_generators = len(generator_IDs)

    ### load the data
    data_folders = [data_dir.format(gen_id) for gen_id in generator_IDs for data_dir in config['data_dirs']]
    for data_folder in data_folders:
        if not os.path.isdir(data_folder):
            print_error('{}: {}: no such directory.'.format(progname, data_folder))
            sys.exit(1)

    inertia = {}
    for key in ('training', 'test'):
        inertia[key] = np.sort([float(re.findall('[0-9]+\.[0-9]*', f)[-1]) \
                                for f in glob.glob(data_folders[0] + '/*' + key + '*.h5')])

    var_names = [var_name.format(gen_id) for gen_id in generator_IDs for var_name in config['var_names']]
    try:
        max_block_size = config['max_block_size']
    except:
        max_block_size = np.inf
    time, x, y = load_data(data_folders, generator_IDs, inertia, var_names, max_block_size, use_tf = False, dtype = np.float32)
    N_vars, N_training_traces, N_samples = x['training'].shape

    ### normalize the data
    x_train_mean = np.mean(x['training'], axis=(1, 2))
    x_train_std = np.std(x['training'], axis=(1, 2))
    for key in x:
        for i, (m,s) in enumerate(zip(x_train_mean, x_train_std)):
            x[key][i] = (x[key][i] - m) / s

    ### store all the parameters
    parameters = config.copy()
    parameters['seed'] = seed
    parameters['N_training_traces'] = N_training_traces
    parameters['N_samples'] = N_samples
    parameters['x_train_mean'] = x_train_mean
    parameters['x_train_std'] = x_train_std
    output_path = args.output_dir + '/random_forest/' + experiment_key
    parameters['output_path'] = output_path
    print(f'Number of training traces: {N_training_traces}')

    ### reshape the data to agree with the forest inputs to the "fit" method
    for key in x:
        tmp = np.transpose(x[key], axes=(1,2,0))
        x[key] = np.reshape(tmp, [tmp.shape[0], N_samples * N_vars], order='F')
        y[key] = np.squeeze(y[key])

    if log_to_comet:
        # add a bunch of tags to the experiment
        experiment.add_tag('random_forest')
        experiment.add_tag('_'.join([f'G{gen_id}' for gen_id in config['generator_IDs']]))
        D = int(re.findall('D=\d', config['data_dirs'][0])[0].split('=')[1])
        DZA = float(re.findall('DZA=\d+.\d+', config['data_dirs'][0])[0].split('=')[1])
        experiment.add_tag(f'DZA={DZA:g}')
        experiment.add_tag(f'D={D:d}')
        
    ### build the random forest
    forest = RandomForestRegressor(n_estimators = config['n_estimators'], \
                                   criterion = config['criterion'].lower(),
                                   max_depth = config['max_depth'],
                                   random_state = rnd_state,
                                   n_jobs = config['n_jobs'] if 'n_jobs' in config else 2,
                                   warm_start = True,
                                   verbose = 3)

    ### train the forest
    forest.fit(x['training'], y['training'])
    forest_params = forest.get_params(deep=True)
    
    ### compute the forest prediction on the test set
    y['prediction'] = forest.predict(x['test'])

    if N_generators == 1:
        for key in y:
            y[key] = y[key][:, np.newaxis]

    ### compute the mean absolute percentage error on the CNN prediction
    mape_prediction = np.mean(np.abs((y['test'] - y['prediction']) / y['test']), axis=0) * 100

    for generator_ID, mape in zip(generator_IDs, mape_prediction):
        print(f'MAPE on CNN prediction for generator {generator_ID} ... {mape:.2f}%')
    test_results = {'y_test': y['test'], 'y_prediction': y['prediction'], 'mape_prediction': mape_prediction}

    os.makedirs(output_path)

    pickle.dump(forest, open(output_path + '/forest.pkl', 'wb'))
    pickle.dump(parameters, open(output_path + '/parameters.pkl', 'wb'))
    pickle.dump(forest_params, open(output_path + '/forest_parameters.pkl', 'wb'))
    pickle.dump(test_results, open(output_path + '/test_results.pkl', 'wb'))

    if log_to_comet:
        experiment.log_parameters(parameters)
        experiment.log_parameters(forest_params)
        experiment.log_metric('mape_prediction', mape_prediction)

    ### plot a summary figure
    cols = N_generators
    fig,ax = plt.subplots(1, cols, figsize=(3 * cols, 3), squeeze=False)
    ax = ax[0,:]
    ### plot the results obtained with the forest
    block_size = y['test'].shape[0] // N_generators
    y_max = np.max(y['training'], axis=0)
    y_min = np.min(y['training'], axis=0)
    for i in range(N_generators):
        limits = [y_min[i], y_max[i]+1]
        ax[i].plot(limits, limits, 'g--')
        idx = np.arange(i * block_size, (i+1) * block_size)
        ax[i].plot(y['test'][idx, i], y['prediction'][idx, i], 'o', color=[1,.7,1], \
                   markersize=4, markerfacecolor='w', markeredgewidth=1)
        for j in range(int(limits[0]), int(limits[1])):
            jdx, = np.where(np.abs(y['test'][idx, i] - (j + 1/3)) < 1e-3)
            m = np.mean(y['prediction'][idx[0] + jdx, i])
            s = np.std(y['prediction'][idx[0] + jdx, i])
            ax[i].plot(j + 1/3 + np.zeros(2), m + s * np.array([-1,1]), 'm-', linewidth=2)
            ax[i].plot(j + 1/3, m, 'ms', markersize=6, markerfacecolor='w', markeredgewidth=2)
        ax[i].axis([1.8, limits[1], 1.8, limits[1]])
        ax[i].set_xlabel('Expected value')
        ax[i].set_title(f'Generator {generator_IDs[i]}')
    ax[0].set_ylabel('Predicted value')
    for a in ax:
        for side in 'right','top':
            a.spines[side].set_visible(False)
    fig.tight_layout()
    if log_to_comet:
        experiment.log_figure('summary', fig)
    plt.savefig(output_path + '/summary.pdf')

