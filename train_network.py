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
import matplotlib.pyplot as plt
from comet_ml import Experiment
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, models

from dlml.data import load_data_areas
from dlml.nn import LEARNING_RATE, build_model, train_model, sigint_handler
from dlml.utils import print_msg, print_warning, print_error

if __name__ == '__main__':

    progname = os.path.basename(sys.argv[0])
    
    parser = arg.ArgumentParser(description = 'Train a network to estimate inertia',
                                formatter_class = arg.ArgumentDefaultsHelpFormatter, \
                                prog = progname)
    parser.add_argument('config_file', type=str, action='store', help='configuration file')
    parser.add_argument('-o', '--output-dir',  default='experiments',  type=str, help='output directory')
    parser.add_argument('--area-measure',  default=None,  type=str, help='area measure (overrides value in configuration file)')
    parser.add_argument('--max-cores',  default=None,  type=int, help='maximum number of cores to be used by Keras)')
    parser.add_argument('--no-comet', action='store_true', help='do not use CometML to log the experiment')
    args = parser.parse_args(args=sys.argv[1:])

    config_file = args.config_file
    if not os.path.isfile(config_file):
        print_error('{}: {}: no such file.'.format(progname, config_file))
        sys.exit(1)
    config = json.load(open(config_file, 'r'))

    with open('/dev/urandom', 'rb') as fid:
        seed = int.from_bytes(fid.read(4), 'little')
    tf.random.set_seed(seed)
    print_msg('Seed: {}'.format(seed))

    if args.area_measure is not None:
        config['area_measure'] = args.area_measure
        print_msg(f'Setting area measure equal to "{args.area_measure}".')

    if args.max_cores is not None:
        config['max_cores'] = args.max_cores
    if 'max_cores' in config and config['max_cores'] is not None:
        max_cores = config['max_cores']
        if max_cores > 0:
            tf.config.threading.set_inter_op_parallelism_threads(max_cores)
            tf.config.threading.set_intra_op_parallelism_threads(max_cores)
            print_msg(f'Maximum number of cores set to {max_cores}.')
        else:
            print_warning('Maximum number of cores must be positive.')

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

    ### an entity can either be an area or a generator
    if 'generator_IDs' in config:
        entity_IDs = config['generator_IDs']
        entity_name = 'generator'
    elif 'area_IDs' in config:
        entity_IDs = config['area_IDs']
        entity_name = 'area'
    else:
        print('One of "area_IDs" or "generator_IDs" must be present in the configuration file.')
        sys.exit(1)

    N_entities = len(entity_IDs)

    ### load the data
    data_folders = [data_dir.format(ntt_id) for ntt_id in entity_IDs for data_dir in config['data_dirs']]
    for data_folder in data_folders:
        if not os.path.isdir(data_folder):
            print_error('{}: {}: no such directory.'.format(progname, data_folder))
            sys.exit(1)

    data_files = {}
    for key in 'training', 'test', 'validation':
        all_files = [sorted(glob.glob(os.path.join(data_folder, '*' + key + '_set.h5'))) for data_folder in data_folders]
        data_files[key] = [item for sublist in all_files for item in sublist]

    var_names = config['var_names']

    use_fft = config['use_fft'] if 'use_fft' in config else False
    if entity_name == 'area':
        generators_areas_map = [config['generators_areas_map'][i-1] for i in config['area_IDs_to_learn_inertia']]
        _, x, y = load_data_areas(data_files,
                                  var_names,
                                  generators_areas_map,
                                  config['generators_Pnom'],
                                  config['area_measure'],
                                  trial_dur=config['trial_duration'] if 'trial_duration' in config else 60.,
                                  max_block_size=config['max_block_size'] if 'max_block_size' in config else np.inf,
                                  use_fft=use_fft)
    else:
        raise Exception('This part is not implemented yet')
        # call load_data_generators in deep_utils

    N_vars, N_training_traces, N_samples = x['training'].shape

    # we always compute mean and std of the training set, whether we'll be using them or not
    x_train_mean = np.mean(x['training'], axis=(1, 2))
    x_train_std = np.std(x['training'], axis=(1, 2))
    x_train_min = np.min(x['training'], axis=(1, 2))
    x_train_max = np.max(x['training'], axis=(1, 2))
    if config['normalization'].lower() == 'training_set':
        for key in x:
            if use_fft:
                x[key] = tf.constant([(x[key][i].numpy() - m) / (M - m) for i,(m,M) in enumerate(zip(x_train_min, x_train_max))])
            else:
                x[key] = tf.constant([(x[key][i].numpy() - m) / s for i,(m,s) in enumerate(zip(x_train_mean, x_train_std))])

    if 'learning_rate_schedule' in config and config['learning_rate_schedule']['name'] is not None:
        lr_schedule = config['learning_rate_schedule'][config['learning_rate_schedule']['name']]
        lr_schedule['name'] = config['learning_rate_schedule']['name']
    else:
        lr_schedule = None

    ### store all the parameters
    parameters = config.copy()
    parameters['seed'] = seed
    parameters['N_training_traces'] = N_training_traces
    parameters['N_samples'] = N_samples
    parameters['x_train_mean'] = x_train_mean
    parameters['x_train_std'] = x_train_std
    parameters['x_train_min'] = x_train_min
    parameters['x_train_max'] = x_train_max
    N_epochs = config['N_epochs']
    batch_size = config['batch_size']
    steps_per_epoch = N_training_traces // batch_size
    parameters['steps_per_epoch'] = steps_per_epoch
    output_path = os.path.join(args.output_dir, 'neural_network', experiment_key)
    parameters['output_path'] = output_path
    print(f'Number of training traces: {N_training_traces}')
    print(f'Batch size:                {batch_size}')
    print(f'Steps per epoch:           {steps_per_epoch}')

    ### build the network
    optimizer_pars = config['optimizer'][config['optimizer']['name']]
    optimizer_pars['name'] = config['optimizer']['name']

    model, optimizer, loss = build_model(N_samples,
                                         steps_per_epoch,
                                         var_names,
                                         config['model_arch'],
                                         len(config['area_IDs_to_learn_inertia']),
                                         config['use_multiple_streams'] if 'use_multiple_streams' in config else 0,
                                         config['normalization'],
                                         config['loss_function'],
                                         optimizer_pars,
                                         lr_schedule)

    if config['normalization'].lower() == 'layer':
        for i,layer in enumerate(model.layers):
            if isinstance(layer, layers.experimental.preprocessing.Normalization):
                break
        for j,layer in enumerate(model.layers[i : i + N_vars]):
            layer.adapt(x['training'][j].numpy().flatten())
            w = layer.get_weights()
            print('-' * 35)
            print(f'{var_names[j]}')
            print(f'means = {x_train_mean[j]:12.4e}, {w[0][0]:12.4e}')
            print(f'vars  = {x_train_std[j] ** 2:12.4e}, {w[1][0]:12.4e}')

    model.summary()

    if log_to_comet:
        experiment.log_parameters(parameters)

    ### reshape the data to agree with the network architecture
    if config['model_arch']['N_dims'] == 2:
        for key in x:
            x[key] = tf.transpose(x[key], perm=(1,2,0))

    try:
        cb_pars = []
        for name in config['callbacks']['names']:
            cb_pars.append(config['callbacks'][name])
            cb_pars[-1]['name'] = name
    except:
        cb_pars = None

    if log_to_comet:
        # add a bunch of tags to the experiment
        experiment.add_tag('neural_network')
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
            experiment.add_tag('buses_' + '_'.join(map(str, bus_numbers)))
            experiment.add_tag('lines_' + '_'.join(map(lambda l: f'{l[0]}-{l[1]}', line_numbers)))
        if use_fft:
            experiment.add_tag('fft')
        try:
            experiment.add_tag('streams_arch_{}'.format(config['use_multiple_streams']))
        except:
            experiment.add_tag('streams_arch_0')
        experiment.add_tag(str(config['model_arch']['N_dims']) + 'D_pipeline')
        try:
            D = int(re.findall('D=\d', config['data_dirs'][0])[0].split('=')[1])
            experiment.add_tag(f'D={D:d}')
        except:
            pass
        try:
            DZA = float(re.findall('DZA=\d+.\d+', config['data_dirs'][0])[0].split('=')[1])
            experiment.add_tag(f'DZA={DZA:g}')
        except:
            pass
        if config['normalization'].lower() == 'batch_norm':
            experiment.add_tag('batch_norm')
        elif config['normalization'].lower() == 'layer':
            experiment.add_tag('normalization_layer')
        try:
            experiment.add_tag(config['learning_rate_schedule']['name'].split('_')[0] + '_lr')
        except:
            pass
        if config['model_arch']['preproc_activation'] is None:
            experiment.add_tag('ReLU_none')
        else:
            activation_fun = config['model_arch']['preproc_activation']
            if activation_fun.lower() == 'relu':
                # make sure the spelling is correct
                activation_fun = 'ReLU'
            experiment.add_tag(activation_fun + '_' + config['model_arch']['activation_loc'])
        try:
            for tag in config['comet_experiment_tags']:
                experiment.add_tag(tag)
        except:
            pass

    ### train the network
    signal.signal(signal.SIGINT, sigint_handler)
    history = train_model(model, x, y,
                          N_epochs,
                          batch_size,
                          steps_per_epoch,
                          output_path,
                          experiment,
                          cb_pars,
                          verbose = 1)

    checkpoint_path = os.path.join(output_path, 'checkpoints')
    
    ### find the best model based on the validation loss
    checkpoint_files = glob.glob(os.path.join(checkpoint_path, '*.h5'))
    try:
        val_loss = [float(file[:-3].split('-')[-1]) for file in checkpoint_files]
        best_checkpoint = checkpoint_files[np.argmin(val_loss)]
    except:
        best_checkpoint = checkpoint_files[-1]
    best_model = models.load_model(best_checkpoint)

    ### compute the network prediction on the test set
    if len(model.inputs) == 1:
        y_prediction = best_model.predict(tf.squeeze(x['test']))
    else:
        y_prediction = best_model.predict({var_name: x['test'][i] for i,var_name
                                                      in enumerate(var_names)})

    if isinstance(y_prediction, list):
        y_prediction = np.squeeze(np.array(y_prediction).T)

    ### compute the mean absolute percentage error on the CNN prediction
    y_test = np.squeeze(y['test'].numpy())
    mape_prediction = losses.mean_absolute_percentage_error(y_test.T, y_prediction.T).numpy()
    for ntt_ID, mape in zip(entity_IDs, mape_prediction):
        print(f'MAPE on CNN prediction for {entity_name} {ntt_ID} ... {mape:.2f}%')
    test_results = {'y_test': y_test, 'y_prediction': y_prediction, 'mape_prediction': mape_prediction}

    best_model.save(output_path)
    pickle.dump(parameters, open(os.path.join(output_path, 'parameters.pkl'), 'wb'))
    pickle.dump(test_results, open(os.path.join(output_path, 'test_results.pkl'), 'wb'))
    pickle.dump(history.history, open(os.path.join(output_path, 'history.pkl'), 'wb'))

    ### plot a graph of the network topology
    keras.utils.plot_model(model, os.path.join(output_path, 'network_topology.png'), show_shapes=True, dpi=300)

    if log_to_comet:
        experiment.log_metric('mape_prediction', mape_prediction)
        experiment.log_model('best_model', os.path.join(output_path, 'saved_model.pb'))
        experiment.log_image(os.path.join(output_path, 'network_topology.png'), 'network_topology')

    ### plot a summary figure
    cols = N_entities + 2
    fig,ax = plt.subplots(1, cols, figsize=(3 * cols, 3))
    ### plot the loss as a function of the epoch number
    epochs = np.r_[0 : len(history.history['loss'])] + 1
    ax[0].plot(epochs, history.history['loss'], 'k', lw=1, label='Training')
    ax[0].plot(epochs, history.history['val_loss'], 'r', lw=1, label='Validation')
    ax[0].legend(loc='best')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ### plot the learning rate as a function of the epoch number
    steps = np.r_[0 : len(LEARNING_RATE)] + 1
    ax[1].semilogy(steps / steps_per_epoch, LEARNING_RATE, 'k', lw=1)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Learning rate')
    ### plot the results obtained with the CNN
    block_size = y['test'].shape[0] // N_entities
    y_max = np.max(y['training'], axis=0)
    y_min = np.min(y['training'], axis=0)
    for i in range(N_entities):
        dy = (y_max[i] - y_min[i]) * 0.05
        limits = [y_min[i] - dy, y_max[i] + dy]
        ax[i+2].plot(limits, limits, 'g--')
        ax[i+2].plot(y['test'][i * block_size : (i+1) * block_size, i], \
                     y_prediction[i * block_size : (i+1) * block_size, i], 'o', \
                     color=[1,.7,1], markersize=4, markerfacecolor='w', markeredgewidth=1)
        for y_target in np.unique(y['test'][i * block_size : (i+1) * block_size, i]):
            idx, = np.where(np.abs(y['test'][i * block_size : (i+1) * block_size, i] - y_target) < 1e-3)
            m = np.mean(y_prediction[idx + i * block_size, i])
            s = np.std(y_prediction[idx + i * block_size, i])
            ax[i+2].plot(y_target + np.zeros(2), m + s * np.array([-1,1]), 'm-', linewidth=2)
            ax[i+2].plot(y_target, m, 'ms', markersize=6, markerfacecolor='w', markeredgewidth=2)
        ax[i+2].axis([limits[0] - dy, limits[1] + dy, limits[0] - dy, limits[1] + dy])
        ax[i+2].set_xlabel('Expected value')
        ax[i+2].set_title(f'{entity_name.capitalize()} {entity_IDs[i]}')
    ax[2].set_ylabel('Predicted value')
    for a in ax:
        for side in 'right','top':
            a.spines[side].set_visible(False)
    fig.tight_layout()
    if log_to_comet:
        experiment.log_figure('summary', fig)
    plt.savefig(os.path.join(output_path, 'summary.pdf'))

