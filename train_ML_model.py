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
import matplotlib.pyplot as plt
from comet_ml import Experiment

from dlml.data import load_data_areas
from dlml.utils import print_msg, print_warning, print_error

def main(progname, args, experiment=None):

    parser = arg.ArgumentParser(description = 'Train a ML model to estimate inertia',
                                formatter_class = arg.ArgumentDefaultsHelpFormatter, \
                                prog = progname)
    parser.add_argument('config_file', type=str, action='store', help='configuration file')
    parser.add_argument('-o', '--output-dir',  default='experiments/ML',  type=str, help='output directory')
    parser.add_argument('--area-measure',  default=None,  type=str, help='area measure (overrides value in configuration file)')
    parser.add_argument('--no-comet', action='store_true', help='do not use CometML to log the experiment')
    parser.add_argument('--save-model', action='store_true', help='save full model (potentially large)')
    args = parser.parse_args(args=args)

    config_file = args.config_file
    if not os.path.isfile(config_file):
        print_error('{}: {}: no such file.'.format(progname, config_file))
        sys.exit(1)
    config = json.load(open(config_file, 'r'))

    model = config['model']
    if model['name'].lower() == 'linear':
        model_name = 'linear'
    elif model['name'].lower() == 'ridge':
        model_name = 'ridge'
    elif model['name'].lower() in ('bayes', 'bayesian_ridge'):
        model_name = 'bayesian_ridge'
    elif model['name'].lower() == 'sgd':
        model_name = 'SGD'
    elif model['name'].lower() == 'kernel_ridge':
        model_name = 'kernel_ridge'
    elif model['name'].lower() in ('svm', 'svr'):
        model_name = 'SVR'
    elif model['name'].lower() in ('neighbors', 'nearest_neighbors'):
        model_name = 'nearest_neighbors'
    elif model['name'].lower() == 'gaussian_processes':
        model_name = 'gaussian_processes'
    elif model['name'].lower() == 'decision_trees':
        model_name = 'decision_trees'
    elif model['name'].lower() in ('forest', 'random_forest'):
        model_name = 'random_forest'
    elif model['name'].lower() in ('mlp', 'multi_layer_perceptron', 'perceptron'):
        model_name = 'MLP'
    else:
        print('Unknown model name: "{}".'.format(model['name']))
        sys.exit(1)
        
    with open('/dev/urandom', 'rb') as fid:
        seed = int.from_bytes(fid.read(4), 'little')
    rnd_state = RandomState(MT19937(SeedSequence(seed)))
    print_msg('Seed: {}'.format(seed))

    if args.area_measure is not None:
        config['area_measure'] = args.area_measure
        print_msg(f'Setting area measure equal to "{args.area_measure}".')

    log_to_comet = not args.no_comet
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
    area_IDs = config['area_IDs']
    N_areas = len(area_IDs)
    if N_areas > 1:
        print('This script only works with 1 area')
        sys.exit(1)

    ### load the data
    data_folders = [data_dir.format(area_ID) for area_ID in area_IDs for data_dir in config['data_dirs']]
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
    normalization = 'minmax' if use_fft else config['normalization']
    if normalization not in ('minmax','z-score','zscore',None):
        print('Unknown normalization mode: `{}`.'.format(normalization))
        sys.exit(1)
    generators_areas_map = [config['generators_areas_map'][i-1] for i in config['area_IDs_to_learn_inertia']]
    t, x, y = load_data_areas(data_files,
                              var_names,
                              generators_areas_map,
                              config['generators_Pnom'],
                              config['area_measure'],
                              trial_dur=config['trial_duration'] if 'trial_duration' in config else 60.,
                              max_block_size=config['max_block_size'] if 'max_block_size' in config else np.inf,
                              use_fft=use_fft, use_tf=False, verbose=True)
    if use_fft:
        freq = t
        sampling_rate = None
    else:
        sampling_rate = 1 / np.diff(t[:2])[0]
        print_msg(f'Sampling rate: {sampling_rate:g} Hz.')

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
    if not use_fft:
        for i,var_name in enumerate(var_names):
            print(f'{var_name} -> {x_train_mean[i]:g} +- {x_train_std[i]:g}, range: [{x_train_min[i]:g},{x_train_max[i]:g}]')
    for key in x:
        for i, (mu,sigma,M,m) in enumerate(zip(x_train_mean, x_train_std, x_train_max, x_train_min)):
            if normalization == 'minmax':
                x[key][i] = (x[key][i] - m) / (M - m)
            elif normalization in ('z-score','zscore'):
                x[key][i] = (x[key][i] - mu) / sigma

    ### store all the parameters
    parameters = config.copy()
    parameters['seed'] = seed
    parameters['N_training_traces'] = N_training_traces
    parameters['N_samples'] = N_samples
    parameters['x_train_mean'] = x_train_mean
    parameters['x_train_std'] = x_train_std
    parameters['x_train_min'] = x_train_min
    parameters['x_train_max'] = x_train_max
    output_path = os.path.join(args.output_dir, model_name, experiment_key)
    parameters['output_path'] = output_path
    print_msg(f'Number of training traces: {N_training_traces}')
    if log_to_comet:
        experiment.log_parameters(parameters)

    for key in x:
        # x[key] has shape (N_vars, N_traces, N_samples): we need to reshape it
        # in such a way that it becomes (N_traces, N_samples * N_vars), i.e.,
        # the i-th row of the new array will contain a concatenation of the i-th
        # N_samples for each of the N_vars variables
        tmp1 = np.transpose(x[key], axes=(1,2,0))
        tmp2 = np.reshape(tmp1, [tmp1.shape[0], N_samples * N_vars], order='F')
        x[key] = tmp2
        # the y vector must be 1D
        if y[key].shape[1] > 1:
            print('We can only deal with 1D output')
            sys.exit(1)
        y[key] = y[key].squeeze()

    if log_to_comet:
        # add a bunch of tags to the experiment
        experiment.add_tag(model_name)
        experiment.add_tag('trial_dur_{:.0f}'.format(config['trial_duration']))
        experiment.add_tag('area_measure_' + config['area_measure'])
        if 'IEEE39' in config['data_dirs'][0]:
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
        if use_fft:
            experiment.add_tag('fft')
        try:
            for tag in config['comet_experiment_tags']:
                experiment.add_tag(tag)
        except:
            pass


    if model_name == 'linear':
        from sklearn.linear_model import LinearRegression
        regr = LinearRegression(n_jobs = model['n_jobs'] if 'n_jobs' in model else -1)
    if model_name == 'ridge':
        from sklearn.linear_model import Ridge
        regr = Ridge(random_state=rnd_state)
    elif model_name == 'bayesian_ridge':
        from sklearn.linear_model import BayesianRidge
        regr = BayesianRidge(verbose=True)
    elif model_name == 'SGD':
        from sklearn.linear_model import SGDRegressor
        regr = SGDRegressor(loss = model['loss'],
                            max_iter = model['max_iter'],
                            tol = model['tol'],
                            random_state = rnd_state,
                            early_stopping = model['early_stopping'],
                            n_iter_no_change = model['n_iter_no_change'],
                            verbose = 3)
    elif model_name == 'kernel_ridge':
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS
        kernel_name = model['kernel'] if 'kernel' in model else 'rbf'
        if kernel_name not in PAIRWISE_KERNEL_FUNCTIONS:
            print('Unknown kernel: {}.'.format(kernel_name))
            sys.exit(1)
        regr = KernelRidge(alpha = model['alpha'],
                           kernel = kernel_name,
                           degree = model['degree'] if 'degree' in model else 3)
    elif model_name == 'SVR':
        from sklearn.svm import NuSVR
        regr = NuSVR(nu = model['nu'],
                     C = model['C'],
                     kernel = model['kernel'],
                     tol = model['tol'],
                     cache_size = 1000,
                     verbose = True,
                     max_iter = model['max_iter'] if 'max_iter' in model else -1)
    elif model_name == 'nearest_neighbors':
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.metrics.pairwise import distance_metrics
        import scipy
        from tslearn import metrics as tslearn_metrics
        metric_name = model['metric'] if 'metric' in model else 'minkowski'
        if metric_name in distance_metrics() or hasattr(scipy.spatial.distance, metric_name):
            metric = metric_name
        elif hasattr(tslearn_metrics, metric_name):
            metric = getattr(tslearn_metrics, metric_name)
        else:
            print('Unknown metric: {}.'.format(metric_name))
            sys.exit(1)
        regr = KNeighborsRegressor(n_neighbors = model['n_neighbors'],
                                   weights = model['weights'],
                                   algorithm = model['algorithm'],
                                   metric = metric,
                                   n_jobs = model['n_jobs'] if 'n_jobs' in model else -1)
    elif model_name == 'gaussian_processes':
        from sklearn.gaussian_process import GaussianProcessRegressor
        regr = GaussianProcessRegressor()
    elif model_name == 'decision_trees':
        from sklearn.tree import DecisionTreeRegressor
        regr = DecisionTreeRegressor(criterion = model['criterion'].lower(),
                                     max_depth = model['max_depth'],
                                     random_state = rnd_state)
    elif model_name == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        max_samples = model['max_samples']
        if max_samples is not None and N_training_traces < max_samples:
            max_samples = N_training_traces
        regr = RandomForestRegressor(n_estimators = model['n_estimators'], \
                                     criterion = model['criterion'].lower(),
                                     max_depth = model['max_depth'],
                                     random_state = rnd_state,
                                     n_jobs = model['n_jobs'] if 'n_jobs' in model else -1,
                                     warm_start = True,
                                     bootstrap = True,
                                     max_samples = max_samples,
                                     verbose = 3)
    elif model_name == 'MLP':
        from sklearn.neural_network import MLPRegressor
        regr = MLPRegressor(hidden_layer_sizes = model['hidden_layer_sizes'],
                            max_iter = model['max_iter'],
                            early_stopping = model['early_stopping'],
                            n_iter_no_change = model['n_iter_no_change'],
                            random_state = rnd_state,
                            verbose = True)
    print(regr)

    ### train the regressor
    regr.fit(x['training'], y['training'])
    regr_params = regr.get_params(deep=True)

    ### compute the regressor prediction on the test set
    y_prediction = regr.predict(x['test'])

    ### compute the mean absolute percentage error on the CNN prediction
    mape_prediction = np.mean(np.abs((y['test'] - y_prediction) / y['test']), axis=0) * 100
    if np.isscalar(mape_prediction): mape_prediction = [mape_prediction]
    for area_ID, mape in zip(area_IDs, mape_prediction):
        print_msg(f'MAPE on CNN prediction for area {area_ID} ... {mape:.2f}%')
    test_results = {'y_test': y['test'], 'y_prediction': y_prediction, 'mape_prediction': mape_prediction}

    os.makedirs(output_path)

    pickle.dump(parameters, open(output_path + '/parameters.pkl', 'wb'))
    pickle.dump(regr_params, open(output_path + '/regressor_parameters.pkl', 'wb'))
    pickle.dump(test_results, open(output_path + '/test_results.pkl', 'wb'))
    if args.save_model:
        pickle.dump(regr, open(output_path + '/model.pkl', 'wb'))

    if log_to_comet:
        experiment.log_parameters(parameters)
        experiment.log_parameters(regr_params)
        experiment.log_metric('mape_prediction', mape_prediction)

    ### plot a summary figure
    fig,ax = plt.subplots(1, 1, figsize=(3, 3))
    y_max_train = np.max(y['training'])
    y_min_train = np.min(y['training'])
    y_max_test = np.max(y['test'])
    y_min_test = np.min(y['test'])
    y_max_pred = np.max(y_prediction)
    y_min_pred = np.min(y_prediction)
    y_max = max(y_max_train, y_max_test, y_max_pred)
    y_min = min(y_min_train, y_min_test, y_min_pred)
    dy = (y_max - y_min) * 0.05
    limits = [y_min - dy, y_max + dy]
    ax.plot(limits, limits, 'g--')
    ax.plot(y['test'], y_prediction, 'o', color=[1,.7,1], markersize=4, markerfacecolor='w', markeredgewidth=1)
    for y_target in np.unique(y['test']):
        idx, = np.where(np.abs(y['test'] - y_target) < 1e-3)
        m = np.mean(y_prediction[idx])
        s = np.std(y_prediction[idx])
        ax.plot(y_target + np.zeros(2), m + s * np.array([-1,1]), 'm-', linewidth=2)
        ax.plot(y_target, m, 'ms', markersize=6, markerfacecolor='w', markeredgewidth=2)
    ax.axis([limits[0] - dy, limits[1] + dy, limits[0] - dy, limits[1] + dy])
    ax.set_xlabel('Expected value')
    #ax.set_title(f'{entity_name.capitalize()} {entity_IDs[i]}')
    ax.set_ylabel('Predicted value')
    for side in 'right','top':
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    if log_to_comet:
        experiment.log_figure('summary', fig)
    plt.savefig(os.path.join(output_path, 'summary.pdf'))

    return output_path

if __name__ == '__main__':
    main(progname=os.path.basename(sys.argv[0]), args=sys.argv[1:])
