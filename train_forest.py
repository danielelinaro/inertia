import os
import sys
import re
import glob
import json
import argparse as arg
from time import strftime, localtime
import pickle
import signal

import numpy as np
from numpy.random import RandomState, SeedSequence, MT19937
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from comet_ml import Experiment

from dlml.data import load_data_areas
from dlml.utils import print_msg, print_warning, print_error

def main(progname, args, experiment=None):

    parser = arg.ArgumentParser(description = 'Train a random forest to estimate inertia',
                                formatter_class = arg.ArgumentDefaultsHelpFormatter, \
                                prog = progname)
    parser.add_argument('config_file', type=str, action='store', help='configuration file')
    parser.add_argument('-o', '--output-dir',  default='experiments',  type=str, help='output directory')
    parser.add_argument('--n-jobs',  default=-1,  type=int, help='number of parallel jobs to run during training')
    parser.add_argument('--area-measure',  default=None,  type=str, help='area measure (overrides value in configuration file)')
    parser.add_argument('--no-comet', action='store_true', help='do not use CometML to log the experiment')
    args = parser.parse_args(args=args)

    config_file = args.config_file
    if not os.path.isfile(config_file):
        print_error('{}: {}: no such file.'.format(progname, config_file))
        sys.exit(1)
    config = json.load(open(config_file, 'r'))

    with open('/dev/urandom', 'rb') as fid:
        seed = int.from_bytes(fid.read(4), 'little')
    rnd_state = RandomState(MT19937(SeedSequence(seed)))
    print_msg('Seed: {}'.format(seed))

    if args.n_jobs is not None:
        config['n_jobs'] = args.n_jobs

    if args.area_measure is not None:
        config['area_measure'] = args.area_measure
        print_msg(f'Setting area measure equal to "{args.area_measure}".')

    log_to_comet = not args.no_comet and False
    if not log_to_comet and experiment is not None:
        print_warning('Ignoring the --no-comet option since the `experiment` argument is not None')
        log_to_comet = True
    
    if log_to_comet:
        ### create a CometML experiment
        if experiment is None:
            experiment = Experiment(
                api_key = os.environ['COMET_API_KEY'],
                project_name = 'inertia',
                workspace = 'danielelinaro'
            )
        experiment_key = experiment.get_key()
    else:
        experiment = None
        experiment_key = strftime('%Y%m%d-%H%M%S', localtime())

    ### an entity can either be an area or a generator
    if 'generator_IDs' in config:
        entity_IDs = config['generator_IDs']
        entity_name = 'generator'
    elif 'area_IDs' in config:
        entity_IDs = config['area_IDs']
        entity_name = 'area'
    else:
        print_error('One of "area_IDs" or "generator_IDs" must be present in the configuration file.')
        sys.exit(1)

    N_entities = len(entity_IDs)

    ### load the data
    data_folders = [data_dir.format(ntt_id) for ntt_id in entity_IDs for data_dir in config['data_dirs']]
    for data_folder in data_folders:
        if not os.path.isdir(data_folder):
            print_error('{}: {}: no such directory.'.format(progname, data_folder))
            sys.exit(1)

    data_files = {}
    for key in 'training', 'test':
        all_files = [sorted(glob.glob(os.path.join(data_folder, '*' + key + '_set.h5'))) for data_folder in data_folders]
        data_files[key] = [item for sublist in all_files for item in sublist]

    var_names = config['var_names']

    use_fft = config['use_fft'] if 'use_fft' in config else False
    if entity_name == 'area':
        generators_areas_map = [config['generators_areas_map'][i-1] for i in config['area_IDs_to_learn_inertia']]
        t, x, y = load_data_areas(data_files,
                                  var_names,
                                  generators_areas_map,
                                  config['generators_Pnom'],
                                  config['area_measure'],
                                  trial_dur=config['trial_duration'] if 'trial_duration' in config else 60.,
                                  F0=config['F0'] if 'F0' in config else 50.,
                                  max_block_size=config['max_block_size'] if 'max_block_size' in config else np.inf,
                                  use_fft=use_fft, use_tf=False, verbose=True)
        if use_fft:
            freq = t
            sampling_rate = None
        else:
            sampling_rate = 1 / np.diff(t[:2])[0]
            print_msg(f'Sampling rate: {sampling_rate:g} Hz.')
    else:
        raise Exception('This part is not implemented yet')
        # call load_data_generators in deep_utils

    N_vars, N_training_traces, N_samples = x['training'].shape

    group = config['group'] if 'group' in config else 1
    if not isinstance(group, int):
        raise Exception('Not implemented yet')
    if group > 1:
        for key in y:
            idx = np.array([np.where(y[key] == val)[0] for val in np.unique(y[key])])
            n_groups = idx.shape[0]
            tmp = y[key].copy()
            for i in range(0, n_groups, group):
                start, stop = i, i + group
                jdx = np.sort(np.concatenate(idx[start:stop]))
                means = tmp[jdx,:].mean(axis=0)
                tmp[jdx,:] = np.tile(means, [jdx.size, 1])
            y[key] = tmp

    ### normalize the data
    x_train_mean = np.mean(x['training'], axis=(1, 2))
    x_train_std = np.std(x['training'], axis=(1, 2))
    x_train_min = np.min(x['training'], axis=(1, 2))
    x_train_max = np.max(x['training'], axis=(1, 2))
    for i,var_name in enumerate(var_names):
        print(f'{var_name} -> {x_train_mean[i]:g} +- {x_train_std[i]:g}, range: [{x_train_min[i]:g},{x_train_max[i]:g}]')
    if config['normalization'].lower() == 'training_set':
        for key in x:
            for i, (m,s) in enumerate(zip(x_train_mean, x_train_std)):
                if use_fft:
                    x[key][i] = (x[key][i] - m) / (M - m)
                else:
                    x[key][i] = (x[key][i] - m) / s

    ### store all the parameters
    parameters = config.copy()
    parameters['seed'] = seed
    parameters['N_training_traces'] = N_training_traces
    parameters['N_samples'] = N_samples
    parameters['x_train_mean'] = x_train_mean
    parameters['x_train_std'] = x_train_std
    parameters['x_train_min'] = x_train_min
    parameters['x_train_max'] = x_train_max
    output_path = os.path.join(args.output_dir, 'random_forest', experiment_key)
    parameters['output_path'] = output_path
    print_msg(f'Number of training traces: {N_training_traces}')
    if log_to_comet:
        experiment.log_parameters(parameters)

    ### reshape the data to agree with the forest inputs to the "fit" method
    for key in x:
        tmp = np.transpose(x[key], axes=(1,2,0))
        x[key] = np.reshape(tmp, [tmp.shape[0], N_samples * N_vars], order='F')

    if log_to_comet:
        # add a bunch of tags to the experiment
        experiment.add_tag('random_forest')
        experiment.add_tag('trial_dur_{:.0f}'.format(config['trial_duration']))
        experiment.add_tag('area_measure_' + config['area_measure'])
        if 'IEEE14' in config['data_dirs'][0]:
            experiment.add_tag('IEEE14')
            experiment.add_tag('_'.join([f'G{gen_id}' for gen_id in config['generator_IDs']]))
        elif 'two-area' in config['data_dirs'][0]:
            experiment.add_tag('two-area')
            if 'area_IDs' in config:
                experiment.add_tag('_'.join([f'area{area_id}' for area_id in config['area_IDs']]))
            else:
                experiment.add_tag('_'.join([f'G{gen_id}' for gen_id in config['generator_IDs']]))
        elif 'IEEE39' in config['data_dirs'][0]:
            bus_numbers = []
            line_numbers = []
            for var_name in config['var_names']:
                if 'bus' in var_name:
                    tmp = int(re.findall('\d+', var_name)[0])
                    if tmp not in bus_numbers:
                        bus_numbers.append(tmp)
                elif 'line' in var_name:
                    tmp = list(map(int, re.findall('\d+', var_name)))
                    if tmp not in line_numbers:
                        line_numbers.append(tmp)
            experiment.add_tag('IEEE39')
            experiment.add_tag('_'.join([f'area{area_id}' for area_id in config['area_IDs_to_learn_inertia']]))
            if len(bus_numbers) > 0: experiment.add_tag('buses_' + '-'.join(map(str, bus_numbers)))
            if len(line_numbers) > 0: experiment.add_tag('lines_' + '-'.join(map(lambda l: f'{l[0]}-{l[1]}', line_numbers)))
        if binary_classification:
            experiment.add_tag('binary_classification')
        elif low_high:
            experiment.add_tag('low_high_prediction')
        if use_fft:
            experiment.add_tag('fft')
        try:
            for tag in config['comet_experiment_tags']:
                experiment.add_tag(tag)
        except:
            pass

    ### build the random forest
    forest = RandomForestRegressor(n_estimators = config['n_estimators'], \
                                   criterion = config['criterion'].lower(),
                                   max_depth = config['max_depth'],
                                   random_state = rnd_state,
                                   n_jobs = config['n_jobs'],
                                   warm_start = True,
                                   bootstrap = True,
                                   max_samples = config['max_samples'],
                                   verbose = 3)

    ### train the forest
    forest.fit(x['training'], y['training'])
    forest_params = forest.get_params(deep=True)

    ### compute the forest prediction on the test set
    y_prediction = forest.predict(x['test'])
    if len(y_prediction.shape):
        y_prediction = y_prediction[:, np.newaxis]

    ### compute the mean absolute percentage error on the CNN prediction
    mape_prediction = np.mean(np.abs((y['test'] - y_prediction) / y['test']), axis=0) * 100
    if np.isscalar(mape_prediction): mape_prediction = [mape_prediction]
    for ntt_ID, mape in zip(entity_IDs, mape_prediction):
        print_msg(f'MAPE on CNN prediction for {entity_name} {ntt_ID} ... {mape:.2f}%')
    test_results = {'y_test': y['test'], 'y_prediction': y_prediction, 'mape_prediction': mape_prediction}

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
    cols = N_entities
    fig,ax = plt.subplots(1, cols, figsize=(3*cols, 3))
    if cols == 1:
        ax = [ax]
    block_size = y['test'].shape[0] // N_entities
    y_max_train = np.max(y['training'], axis=0)
    y_min_train = np.min(y['training'], axis=0)
    y_max_test = np.max(y['test'], axis=0)
    y_min_test = np.min(y['test'], axis=0)
    y_max_pred = np.max(y_prediction, axis=0)
    y_min_pred = np.min(y_prediction, axis=0)
    y_max = np.max(np.concatenate((y_max_train[np.newaxis,:], y_max_test[np.newaxis,:], y_max_pred[np.newaxis,:]), axis=0), axis=0)
    y_min = np.min(np.concatenate((y_min_train[np.newaxis,:], y_min_test[np.newaxis,:], y_min_pred[np.newaxis,:]), axis=0), axis=0)
    for i in range(N_entities):
        dy = (y_max[i] - y_min[i]) * 0.05
        limits = [y_min[i] - dy, y_max[i] + dy]
        ax[i].plot(limits, limits, 'g--')
        ax[i].plot(y['test'][i * block_size : (i+1) * block_size, i], \
                     y_prediction[i * block_size : (i+1) * block_size, i], 'o', \
                     color=[1,.7,1], markersize=4, markerfacecolor='w', markeredgewidth=1)
        for y_target in np.unique(y['test'][i * block_size : (i+1) * block_size, i]):
            idx, = np.where(np.abs(y['test'][i * block_size : (i+1) * block_size, i] - y_target) < 1e-3)
            m = np.mean(y_prediction[idx + i * block_size, i])
            s = np.std(y_prediction[idx + i * block_size, i])
            ax[i].plot(y_target + np.zeros(2), m + s * np.array([-1,1]), 'm-', linewidth=2)
            ax[i].plot(y_target, m, 'ms', markersize=6, markerfacecolor='w', markeredgewidth=2)
        ax[i].axis([limits[0] - dy, limits[1] + dy, limits[0] - dy, limits[1] + dy])
        ax[i].set_xlabel('Expected value')
        ax[i].set_title(f'{entity_name.capitalize()} {entity_IDs[i]}')
    ax[0].set_ylabel('Predicted value')
    for a in ax:
        for side in 'right','top':
            a.spines[side].set_visible(False)
    fig.tight_layout()
    if log_to_comet:
        experiment.log_figure('summary', fig)
    plt.savefig(os.path.join(output_path, 'summary.pdf'))

    return output_path

if __name__ == '__main__':
    main(progname=os.path.basename(sys.argv[0]), args=sys.argv[1:])
