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


def make_preprocessing_pipeline_1D(N_samples, N_units, kernel_size, activation_fun, activation_loc, input_name):
    if activation_fun is not None:
        if activation_fun.lower() not in ('relu',):
            raise Exception(f'Unknown activation function {activation_fun}')
        if activation_loc is None:
            raise Exception(f'Must specify activation function location')
        elif activation_loc.lower() not in ('after_conv', 'after_pooling'):
            raise Exception('activation_loc must be one of "after_conv" or "after_pooling"')
    inp = keras.Input(shape=(N_samples, 1), name=input_name)
    for N_conv,N_pooling,sz in zip(N_units['conv'], N_units['pooling'], kernel_size):
        try:
            L = layers.Conv1D(N_conv, sz, activation=None)(L)
        except:
            L = layers.Conv1D(N_conv, sz, activation=None)(inp)
        if activation_fun is not None:
            if activation_loc == 'after_conv':
                L = layers.ReLU()(L)
                L = layers.MaxPooling1D(N_pooling)(L)
            else:
                L = layers.MaxPooling1D(N_pooling)(L)
                L = layers.ReLU()(L)
        else:
            L = layers.MaxPooling1D(N_pooling)(L)
    return inp,L


def make_preprocessing_pipeline_2D(N_samples, N_units, kernel_size, activation_fun, activation_loc, input_name):
    if activation_fun is not None:
        if activation_fun.lower() not in ('relu',):
            raise Exception(f'Unknown activation function {activation_fun}')
        if activation_loc is None:
            raise Exception(f'Must specify activation function location')
        elif activation_loc.lower() not in ('after_conv', 'after_pooling'):
            raise Exception('activation_loc must be one of "after_conv" or "after_pooling"')
    inp = keras.Input(shape=(N_samples, 2, 1), name=input_name)
    for N_conv,N_pooling,sz in zip(N_units['conv'], N_units['pooling'], kernel_size):
        try:
            L = layers.Conv2D(N_conv, [sz, 2], padding='same', activation=None)(L)
        except:
            L = layers.Conv2D(N_conv, [sz, 2], padding='same', activation=None)(inp)
        if activation_fun is not None:
            if activation_loc == 'after_conv':
                L = layers.ReLU()(L)
                L = layers.MaxPooling2D([N_pooling, 1])(L)
            else:
                L = layers.MaxPooling2D([N_pooling, 1])(L)
                L = layers.ReLU()(L)
        else:
            L = layers.MaxPooling2D([N_pooling, 1])(L)
    return inp,L


def build_model(N_samples, var_names, model_arch, loss_fun_pars, optimizer_pars):
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

    if N_dims == 1:
        inputs = []
        L = []
        for var_name in var_names:
            inp,lyr = make_preprocessing_pipeline_1D(N_samples, N_units, kernel_size, \
                                                     activation_fun, activation_loc, \
                                                     var_name)
            inputs.append(inp)
            L.append(lyr)
    else:
        inputs,L = make_preprocessing_pipeline_2D(N_samples, N_units, kernel_size, \
                                                  activation_fun, activation_loc, \
                                                  '_'.join(var_names))

    if isinstance(L, list):
        L = layers.concatenate(L)
    L = layers.Flatten()(L)
    for n in N_units['dense']:
        L = layers.Dense(n, activation='relu')(L)
    if model_arch['dropout_coeff'] > 0:
        L = layers.Dropout(model_arch['dropout_coeff'])(L)
    output = layers.Dense(y['training'].shape[1])(L)

    model = keras.Model(inputs=inputs, outputs=output)

    model.compile(optimizer=optimizer, loss=loss)

    return model, optimizer, loss



def train_model(model, x, y,
                N_epochs,
                batch_size,
                steps_per_epoch,
                output_dir,
                early_stopping_patience = None,
                learning_rate_schedule = None,
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
    cbs = [checkpoint_cb]

    if early_stopping_patience is not None:
        # create a callback that will stop the optimization if there is no improvement
        early_stop_cb = callbacks.EarlyStopping(monitor = 'val_loss',
                                                patience = early_stopping_patience,
                                                verbose = verbose,
                                                mode = 'min')

        cbs.append(early_stop_cb)
        print_msg('Added a callback for early stopping.')
    else:
        print_warning('Not adding a callback for early stopping.')

    try:

        if learning_rate_schedule['name'] == 'cyclical':
            # According to [1], "experiments show that it often is good to set stepsize equal to
            # 2 − 10 times the number of iterations in an epoch".
            #
            # [1] Smith, L.N., 2017, March.
            #     Cyclical learning rates for training neural networks.
            #     In 2017 IEEE Winter Conference on Applications of Computer Vision (WACV) (pp. 464-472). IEEE.
            #
            schedule = tfa.optimizers.Triangular2CyclicalLearningRate(
                initial_learning_rate = learning_rate_schedule['initial_learning_rate'],
                maximal_learning_rate = learning_rate_schedule['max_learning_rate'],
                step_size = x['training'].shape[0] // batch_size * learning_rate_schedule['factor'])
            lr_scheduler_cb = callbacks.LearningRateScheduler(schedule, verbose)
            cbs.append(lr_scheduler_cb)
            print_msg('Added a callback for cyclical learning rate scheduling.')

        elif learning_rate_schedule['name'] == 'reduce_on_plateau':
            optimizer_lr = model.optimizer.learning_rate.numpy()
            factor = learning_rate_schedule['factor']
            patience = learning_rate_schedule['patience']
            cooldown = learning_rate_schedule['cooldown'] if 'cooldown' \
                       in learning_rate_schedule else patience / 10
            min_lr = learning_rate_schedule['min_learning_rate'] if 'min_learning_rate' \
                     in learning_rate_schedule else optimizer_lr / 1000
            reduce_lr_cb = callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                                       factor = factor,
                                                       patience = patience,
                                                       verbose = verbose,
                                                       mode = 'min',
                                                       cooldown = cooldown,
                                                       min_lr = min_lr)
            cbs.append(reduce_lr_cb)
            print_msg('Added a callback for reducing learning rate on plateaus.')

        else:
            raise Exception('Unknown learning rate scheduling policy: {}'.format(learning_rate_schedule['name']))

    except:
        print_warning('Not adding a callback for learning rate scheduling.')

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
    parser.add_argument('--no-comet', action='store_true', help='do not use CometML to log the experiment')
    args = parser.parse_args(args=sys.argv[1:])

    config_file = args.config_file
    if not os.path.isfile(config_file):
        print_error('{}: {}: no such file.'.format(progname, config_file))
        sys.exit(1)
    config = json.load(open(config_file, 'r'))

    with open('/dev/random', 'rb') as fid:
        seed = int.from_bytes(fid.read(4), 'little')
    tf.random.set_seed(seed)
    print_msg('Seed: {}'.format(seed))

    log_to_comet = not args.no_comet

    if log_to_comet:
        ### create a CometML experiment
        experiment = Experiment(
            api_key = os.environ['COMET_ML_API_KEY'],
            project_name = 'inertia',
            workspace = 'danielelinaro'
        )
        experiment_key = experiment.get_key()
    else:
        experiment_key = strftime('%Y%m%d-%H%M%S', localtime())

    ### generator IDs
    generator_IDs = config['generator_IDs']
    n_generators = len(generator_IDs)

    ### load the data
    data_folders = [data_dir.format(gen_id) for gen_id in generator_IDs for data_dir in config['data_dirs']]
    for data_folder in data_folders:
        if not os.path.isdir(data_folder):
            print_error('{}: {}: no such directory.'.format(progname, data_folder))
            sys.exit(1)

    inertia = {}
    for key in ('training', 'test', 'validation'):
        for ext in ('.h5', '.npz'):
            inertia[key] = np.sort([float(re.findall('[0-9]+\.[0-9]*', f)[-1]) \
                                    for f in glob.glob(data_folders[0] + '/*' + key + '*' + ext)])
            if len(inertia[key]) > 0:
                break

    var_names = [var_name.format(gen_id) for gen_id in generator_IDs for var_name in config['var_names']]
    try:
        max_block_size = config['max_block_size']
    except:
        max_block_size = np.inf
    time, x, y = load_data(data_folders, generator_IDs, inertia, var_names, max_block_size)
    N_vars, N_training_traces, N_samples = x['training'].shape

    ### normalize the data
    x_train_mean = np.mean(x['training'], axis=(1, 2))
    x_train_std = np.std(x['training'], axis=(1, 2))
    for key in x:
        x[key] = tf.constant([(x[key][i].numpy() - m) / s for i,(m,s) in enumerate(zip(x_train_mean, x_train_std))])

    ### build the network
    optimizer_pars = config['optimizer'][config['optimizer']['name']]
    optimizer_pars['name'] = config['optimizer']['name']
    model, optimizer, loss = build_model(N_samples,
                                         var_names,
                                         config['model_arch'],
                                         config['loss_function'],
                                         optimizer_pars)
    model.summary()

    ### store all the parameters
    parameters = config.copy()
    parameters['seed'] = seed
    parameters['N_training_traces'] = N_training_traces
    parameters['N_samples'] = N_samples
    parameters['x_train_mean'] = x_train_mean
    parameters['x_train_std'] = x_train_std
    
    try:
        es_patience = config['early_stopping_patience']
    except:
        es_patience = None
        parameters['early_stopping_patience'] = None

    if 'learning_rate_schedule' in config and \
       config['learning_rate_schedule']['name'].lower() != 'none':
        lr_schedule = config['learning_rate_schedule'][config['learning_rate_schedule']['name']]
        lr_schedule['name'] = config['learning_rate_schedule']['name']
    else:
        lr_schedule = None
        parameters['learning_rate_schedule'] = None

    N_epochs = config['N_epochs']
    batch_size = config['batch_size']
    N_batches = np.ceil(N_training_traces / batch_size)
    parameters['N_batches'] = N_batches
    steps_per_epoch = np.max([N_batches, 100])
    parameters['steps_per_epoch'] = steps_per_epoch
    output_path = args.output_dir + '/' + experiment_key
    parameters['output_path'] = output_path

    if log_to_comet:
        experiment.log_parameters(parameters)

    ### reshape the data to agree with the network architecture
    if config['model_arch']['N_dims'] == 2:
        for key in x:
            x[key] = tf.transpose(x[key], perm=(1,2,0))

    ### train the network
    history = train_model(model, x, y,
                          N_epochs,
                          batch_size,
                          steps_per_epoch,
                          output_path,
                          es_patience,
                          lr_schedule,
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
    for generator_ID, mape in zip(generator_IDs, mape_prediction):
        print(f'MAPE on CNN prediction for generator {generator_ID} ... {mape:.2f}%')
    test_results = {'y_test': y_test, 'y_prediction': y_prediction, 'mape_prediction': mape_prediction}
    
    best_model.save(output_path)
    pickle.dump(parameters, open(output_path + '/parameters.pkl', 'wb'))
    pickle.dump(test_results, open(output_path + '/test_results.pkl', 'wb'))
    pickle.dump(history.history, open(output_path + '/history.pkl', 'wb'))

    if log_to_comet:
        experiment.log_metric('mape_prediction', mape_prediction)
        experiment.log_model('best_model', output_path + '/saved_model.pb')

    ### plot a graph of the network topology
    keras.utils.plot_model(model, output_path + '/network_topology.pdf', show_shapes=True, dpi=300)

    fig,ax = plt.subplots(1, n_generators + 1, figsize=(3 + 3 * n_generators, 3))
    ### plot the loss as a function of the epoch number
    epochs = np.r_[0 : len(history.history['loss'])] + 1
    ax[0].plot(epochs, history.history['loss'], 'k', label='Training')
    ax[0].plot(epochs, history.history['val_loss'], 'r', label='Validation')
    ax[0].legend(loc='best')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ### plot the results obtained with the CNN
    block_size = y['test'].shape[0] // n_generators
    y_max = np.max(y['training'], axis=0)
    y_min = np.min(y['training'], axis=0)
    for i in range(n_generators):
        limits = [y_min[i], y_max[i]+1]
        ax[i+1].plot(limits, limits, 'g--')
        idx = np.arange(i * block_size, (i+1) * block_size)
        ax[i+1].plot(y['test'][i * block_size : (i+1) * block_size, i], \
                     y_prediction[i * block_size : (i+1) * block_size, i], 'o', \
                     color=[1,.7,1], markersize=4, markerfacecolor='w', markeredgewidth=1)
        for j in range(int(limits[0]), int(limits[1])):
            idx, = np.where(np.abs(y['test'][i * block_size : (i+1) * block_size, i] - (j + 1/3)) < 1e-3)
            m = np.mean(y_prediction[idx + i * block_size,i])
            s = np.std(y_prediction[idx + i * block_size,i])
            ax[i+1].plot(j+1/3 + np.zeros(2), m + s * np.array([-1,1]), 'm-', linewidth=2)
            ax[i+1].plot(j+1/3, m, 'ms', markersize=8, markerfacecolor='w', markeredgewidth=2)
        ax[i+1].axis([1.8, limits[1], 1.8, limits[1]])
        ax[i+1].set_xlabel('Expected value')
        ax[i+1].set_title(f'Generator {generator_IDs[i]}')
    ax[1].set_ylabel('Predicted value')
    fig.tight_layout()
    plt.savefig(output_path + '/summary.pdf')

