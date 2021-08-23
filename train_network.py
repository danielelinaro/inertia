import os
import sys
import re
import glob
import json
import argparse as arg
from time import strftime, localtime
import pickle

import numpy as np
import matplotlib.pyplot as plt
from comet_ml import Experiment
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, callbacks, models
import tensorflow_addons as tfa
from deep_utils import *

import colorama as cm
print_error   = lambda msg: print(f'{cm.Fore.RED}'    + msg + f'{cm.Style.RESET_ALL}')
print_warning = lambda msg: print(f'{cm.Fore.YELLOW}' + msg + f'{cm.Style.RESET_ALL}')
print_msg     = lambda msg: print(f'{cm.Fore.GREEN}'  + msg + f'{cm.Style.RESET_ALL}')


LEARNING_RATE = []

class LearningRateCallback(keras.callbacks.Callback):
    def __init__(self, model, experiment = None):
        self.model = model
        self.experiment = experiment
        self.step = 0

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        try:
            lr = self.model.optimizer.learning_rate(self.step).numpy()
        except:
            lr = self.model.optimizer.learning_rate
        LEARNING_RATE.append(lr)
        if self.experiment is not None:
            self.experiment.log_metric('learning_rate', lr, self.step)


def make_preprocessing_pipeline_1D(input_layer, N_units, kernel_size, activation_fun, activation_loc):
    if activation_fun is not None:
        if activation_fun.lower() not in ('relu',):
            raise Exception(f'Unknown activation function {activation_fun}')
        if activation_loc is None:
            raise Exception(f'Must specify activation function location')
        elif activation_loc.lower() not in ('after_conv', 'after_pooling'):
            raise Exception('activation_loc must be one of "after_conv" or "after_pooling"')
    base_name = input_layer.name
    for n,(N_conv,N_pooling,sz) in enumerate(zip(N_units['conv'], N_units['pooling'], kernel_size)):
        conv_lyr_name = base_name + f'_conv{n+1}'
        activ_lyr_name = base_name + f'_relu{n+1}'
        pool_lyr_name = base_name + f'_pool{n+1}'
        try:
            L = layers.Conv1D(N_conv, sz, activation=None, name=conv_lyr_name)(L)
        except:
            L = layers.Conv1D(N_conv, sz, activation=None, name=conv_lyr_name)(input_layer)
        if activation_fun is not None:
            if activation_loc == 'after_conv':
                L = layers.ReLU(name=activ_lyr_name)(L)
                L = layers.MaxPooling1D(N_pooling,  name=pool_lyr_name)(L)
            else:
                L = layers.MaxPooling1D(N_pooling, name=pool_lyr_name)(L)
                L = layers.ReLU(name=activ_lyr_name)(L)
        else:
            L = layers.MaxPooling1D(N_pooling, name=pool_lyr_name)(L)
    return L


def make_preprocessing_pipeline_2D(input_layer, N_units, kernel_size, activation_fun, activation_loc):
    if activation_fun is not None:
        if activation_fun.lower() not in ('relu',):
            raise Exception(f'Unknown activation function {activation_fun}')
        if activation_loc is None:
            raise Exception(f'Must specify activation function location')
        elif activation_loc.lower() not in ('after_conv', 'after_pooling'):
            raise Exception('activation_loc must be one of "after_conv" or "after_pooling"')
    base_name = input_layer.name
    for n,(N_conv,N_pooling,sz) in enumerate(zip(N_units['conv'], N_units['pooling'], kernel_size)):
        conv_lyr_name = base_name + f'_conv{n+1}'
        activ_lyr_name = base_name + f'_relu{n+1}'
        pool_lyr_name = base_name + f'_pool{n+1}'
        try:
            L = layers.Conv2D(N_conv, [sz, 2], padding='same', activation=None, name=conv_lyr_name)(L)
        except:
            L = layers.Conv2D(N_conv, [sz, 2], padding='same', activation=None, name=conv_lyr_name)(input_layer)
        if activation_fun is not None:
            if activation_loc == 'after_conv':
                L = layers.ReLU(name=activ_lyr_name)(L)
                L = layers.MaxPooling2D([N_pooling, 1], name=pool_lyr_name)(L)
            else:
                L = layers.MaxPooling2D([N_pooling, 1], name=pool_lyr_name)(L)
                L = layers.ReLU(name=activ_lyr_name)(L)
        else:
            L = layers.MaxPooling2D([N_pooling, 1], name=pool_lyr_name)(L)
    return L


def build_model(N_samples, steps_per_epoch, var_names, model_arch, \
                normalization_strategy, loss_fun_pars, optimizer_pars, lr_schedule_pars):
    """
    Builds and compiles the model
    
    The basic network topology used here is taken from the following paper:
   
    George, D., & Huerta, E. A. (2018).
    Deep neural networks to enable real-time multimessenger astrophysics.
    Physical Review D, 97(4), 044039. http://doi.org/10.1103/PhysRevD.97.044039
    """

    N_dims = model_arch['N_dims']
    if N_dims not in (1,2):
        raise Exception('The number of dimensions of the data must be either 1 or 2')

    loss_fun_name = loss_fun_pars['name'].lower()
    if loss_fun_name == 'mae':
        loss = losses.MeanAbsoluteError()
    elif loss_fun_name == 'mape':
        loss = losses.MeanAbsolutePercentageError()
    else:
        raise Exception('Unknown loss function: {}.'.format(loss_function))

    if lr_schedule_pars is not None and 'name' in lr_schedule_pars:
        if lr_schedule_pars['name'] == 'cyclical':
            # According to [1], "experiments show that it often is good to set stepsize equal to
            # 2 − 10 times the number of iterations in an epoch".
            #
            # [1] Smith, L.N., 2017, March.
            #     Cyclical learning rates for training neural networks.
            #     In 2017 IEEE Winter Conference on Applications of Computer Vision (WACV) (pp. 464-472). IEEE.
            #
            step_sz = steps_per_epoch * lr_schedule_pars['factor']
            learning_rate = tfa.optimizers.Triangular2CyclicalLearningRate(
                initial_learning_rate = lr_schedule_pars['initial_learning_rate'],
                maximal_learning_rate = lr_schedule_pars['max_learning_rate'],
                step_size = step_sz)
            print_msg(f'Will use cyclical learning rate scheduling with a step size of {step_sz}.')
        elif lr_schedule_pars['name'] == 'exponential_decay':
            initial_learning_rate = lr_schedule_pars['initial_learning_rate']
            decay_steps = lr_schedule_pars['decay_steps']
            decay_rate = lr_schedule_pars['decay_rate']
            learning_rate = optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps, decay_rate)
            print_msg('Will use exponential decay of learning rate.')
        else:
            raise Exception(f'Unknown learning rate schedule: {lr_schedule_pars["name"]}')
    else:
        learning_rate = optimizer_pars['learning_rate']

    optimizer_name = optimizer_pars['name'].lower()
    if optimizer_name == 'sgd':
        momentum = optimizer_pars['momentum'] if 'momentum' in optimizer_pars else 0.
        nesterov = optimizer_pars['nesterov'] if 'nesterov' in optimizer_pars else False
        optimizer = optimizers.SGD(learning_rate, momentum, nesterov)
    elif optimizer_name in ('adam', 'adamax', 'nadam'):
        beta_1 = optimizer_pars['beta_1'] if 'beta_1' in optimizer_pars else 0.9
        beta_2 = optimizer_pars['beta_2'] if 'beta_2' in optimizer_pars else 0.999
        if optimizer_name == 'adam':
            optimizer = optimizers.Adam(learning_rate, beta_1, beta_2)
        elif optimizer_name == 'adamax':
            optimizer = optimizers.Adamax(learning_rate, beta_1, beta_2)
        else:
            optimizer = optimizers.Nadam(learning_rate, beta_1, beta_2)
    elif optimizer_name == 'adagrad':
        initial_accumulator_value = optimizer_pars['initial_accumulator_value'] if \
                                    'initial_accumulator_value' in optimizer_pars else 0.1
        optimizer = optimizers.Adagrad(learning_rate, initial_accumulator_value)
    elif optimizer_name == 'adadelta':
        rho = optimizer_pars['rho'] if 'rho' in optimizer_pars else 0.95
        optimizer = optimizers.Adadelta(learning_rate, rho)
    else:
        raise Exception('Unknown optimizer: {}.'.format(optimizer_name))

    N_units = model_arch['N_units']
    kernel_size = model_arch['kernel_size']
    activation_fun = model_arch['preproc_activation']
    activation_loc = model_arch['activation_loc']

    ### figure out how data should be normalized
    batch_norm = config['normalization'].lower() == 'batch'
    normalization_layer = config['normalization'].lower() == 'layer'

    if N_dims == 1:
        inputs = []
        L = []
        for var_name in var_names:
            input_layer = keras.Input(shape=(N_samples, 1), name=var_name)
            inputs.append(input_layer)
            if batch_norm:
                input_layer = layers.BatchNormalization()(input_layer)
            elif normalization_layer:
                input_layer = layers.experimental.preprocessing.Normalization()(input_layer)
            lyr = make_preprocessing_pipeline_1D(input_layer, N_units, kernel_size, \
                                                 activation_fun, activation_loc)
            L.append(lyr)
    else:
        inputs = keras.Input(shape=(N_samples, 2, 1), name='_'.join(var_names))
        if batch_norm:
            input_layer = layers.BatchNormalization()(inputs)
        elif normalization_layer:
            input_layer = layers.experimental.preprocessing.Normalization()(inputs)
        else:
            input_layer = inputs
        L = make_preprocessing_pipeline_2D(input_layer, N_units, kernel_size, \
                                           activation_fun, activation_loc)

    if isinstance(L, list):
        L = layers.concatenate(L)
    L = layers.Flatten(name='flatten')(L)
    for i,n in enumerate(N_units['dense']):
        L = layers.Dense(n, activation='relu', name=f'fc{i+1}')(L)
    if model_arch['dropout_coeff'] > 0:
        L = layers.Dropout(model_arch['dropout_coeff'], name='dropout')(L)
    output = layers.Dense(y['training'].shape[1], name='predictions')(L)

    model = keras.Model(inputs=inputs, outputs=output)

    model.compile(optimizer=optimizer, loss=loss)

    return model, optimizer, loss


def train_model(model, x, y,
                N_epochs,
                batch_size,
                steps_per_epoch,
                output_dir,
                experiment = None,
                callbacks_pars = None,
                verbose = 1):

    checkpoint_dir = output_dir + '/checkpoints'
    os.makedirs(checkpoint_dir)

    # create a callback that saves the model's weights
    checkpoint_cb = callbacks.ModelCheckpoint(filepath = checkpoint_dir + \
                                              '/weights.{epoch:04d}-{val_loss:.6f}.h5',
                                              save_weights_only = False,
                                              save_best_only = True,
                                              monitor = 'val_loss',
                                              verbose = verbose)
    print_msg('Added callback for saving weights at checkpoint.')

    cbs = [checkpoint_cb, LearningRateCallback(model, experiment)]
    print_msg('Added callback for logging learning rate.')

    try:
        for cb_pars in callbacks_pars:
            if cb_pars['name'] == 'early_stopping':
                # create a callback that will stop the optimization if there is no improvement
                early_stop_cb = callbacks.EarlyStopping(monitor = cb_pars['monitor'],
                                                        patience = cb_pars['patience'],
                                                        verbose = verbose,
                                                        mode = cb_pars['mode'])
                cbs.append(early_stop_cb)
                print_msg('Added callback for early stopping.')
            elif cb_pars['name'] == 'reduce_on_plateau':
                lr_scheduler_cb = callbacks.ReduceLROnPlateau(monitor = cb_pars['monitor'],
                                                              factor = cb_pars['factor'],
                                                              patience = cb_pars['patience'],
                                                              verbose = verbose,
                                                              mode = cb_pars['mode'],
                                                              cooldown = cb_pars['cooldown'],
                                                              min_lr = cb_pars['min_lr'])
                cbs.append(lr_scheduler_cb)
                print_msg('Added callback for reducing learning rate on plateaus.')
            else:
                raise Exception(f'Unknown callback: {cb_pars["name"]}')
    except:
        print_warning('Not adding callbacks.')

    if len(model.inputs) == 1:
        x_training = x['training']
        x_validation = x['validation']
    else:
        input_names = [inp.name.split(':')[0] for inp in model.inputs]
        x_training = {name: x['training'][i] for i,name in enumerate(input_names)}
        x_validation = {name: x['validation'][i] for i,name in enumerate(input_names)}

    return model.fit(x_training,
                     y['training'],
                     epochs = N_epochs,
                     batch_size = batch_size,
                     steps_per_epoch = steps_per_epoch,
                     validation_data = (x_validation, y['validation']),
                     verbose = verbose,
                     callbacks = cbs)



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
        all_files = [glob.glob(data_folder + '/*' + key + '_set.h5') for data_folder in data_folders]
        data_files[key] = [item for sublist in all_files for item in sublist]

    var_names = config['var_names']

    try:
        max_block_size = config['max_block_size']
    except:
        max_block_size = np.inf

    if entity_name == 'area':
        generators_areas_map = [config['generators_areas_map'][i-1] for i in config['area_IDs_to_learn_inertia']]
        time, x, y = load_data_areas(data_files,
                                     var_names,
                                     generators_areas_map,
                                     config['generators_Pnom'],
                                     config['area_measure'],
                                     max_block_size)
    else:
        print('This part is not implemented yet')
        # call load_data_generators in deep_utils
        import ipdb
        ipdb.set_trace()

    N_vars, N_training_traces, N_samples = x['training'].shape

    # we always compute mean and std of the training set, whether we'll be using them or not
    x_train_mean = np.mean(x['training'], axis=(1, 2))
    x_train_std = np.std(x['training'], axis=(1, 2))
    if config['normalization'].lower() == 'training_set':
        for key in x:
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
    N_epochs = config['N_epochs']
    batch_size = config['batch_size']
    steps_per_epoch = N_training_traces // batch_size
    parameters['steps_per_epoch'] = steps_per_epoch
    output_path = args.output_dir + '/neural_network/' + experiment_key
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
            bus_names = np.unique([int(re.findall('\d+', var_name)[0]) for var_name in config['var_names']])
            experiment.add_tag('IEEE39')
            experiment.add_tag('_'.join([f'area{area_id}' for area_id in config['area_IDs_to_learn_inertia']]))
            experiment.add_tag('buses_' + '-'.join([str(bus_name) for bus_name in bus_names]))
        experiment.add_tag(str(config['model_arch']['N_dims']) + 'D_pipeline')
        D = int(re.findall('D=\d', config['data_dirs'][0])[0].split('=')[1])
        DZA = float(re.findall('DZA=\d+.\d+', config['data_dirs'][0])[0].split('=')[1])
        experiment.add_tag(f'DZA={DZA:g}')
        experiment.add_tag(f'D={D:d}')
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
    history = train_model(model, x, y,
                          N_epochs,
                          batch_size,
                          steps_per_epoch,
                          output_path,
                          experiment,
                          cb_pars,
                          verbose = 1)

    checkpoint_path = output_path + '/checkpoints'
    
    ### find the best model based on the validation loss
    checkpoint_files = glob.glob(checkpoint_path + '/*.h5')
    val_loss = [float(file[:-3].split('-')[-1]) for file in checkpoint_files]
    best_checkpoint = checkpoint_files[np.argmin(val_loss)]
    best_model = models.load_model(best_checkpoint)

    ### compute the network prediction on the test set
    if len(model.inputs) == 1:
        y_prediction = best_model.predict(x['test'])
    else:
        y_prediction = best_model.predict({var_name: x['test'][i] for i,var_name
                                                      in enumerate(var_names)})

    ### compute the mean absolute percentage error on the CNN prediction
    y_test = np.squeeze(y['test'].numpy())
    mape_prediction = losses.mean_absolute_percentage_error(y_test.T, y_prediction.T).numpy()
    for ntt_ID, mape in zip(entity_IDs, mape_prediction):
        print(f'MAPE on CNN prediction for {entity_name} {ntt_ID} ... {mape:.2f}%')
    test_results = {'y_test': y_test, 'y_prediction': y_prediction, 'mape_prediction': mape_prediction}

    best_model.save(output_path)
    pickle.dump(parameters, open(output_path + '/parameters.pkl', 'wb'))
    pickle.dump(test_results, open(output_path + '/test_results.pkl', 'wb'))
    pickle.dump(history.history, open(output_path + '/history.pkl', 'wb'))

    ### plot a graph of the network topology
    keras.utils.plot_model(model, output_path + '/network_topology.png', show_shapes=True, dpi=300)

    if log_to_comet:
        experiment.log_metric('mape_prediction', mape_prediction)
        experiment.log_model('best_model', output_path + '/saved_model.pb')
        experiment.log_image(output_path + '/network_topology.png', 'network_topology')

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
    plt.savefig(output_path + '/summary.pdf')

