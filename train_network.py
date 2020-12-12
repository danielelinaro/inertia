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
from deep_utils import *



def build_network(N_samples, depth_level = 1, learning_rate = 1e-4, dropout_coeff = None, loss_function = 'mae', optimizer = 'adam', full_output = False):
    """
    Builds the network
    
    The network topology used here is taken from the following paper:
   
    George, D., & Huerta, E. A. (2018).
    Deep neural networks to enable real-time multimessenger astrophysics.
    Physical Review D, 97(4), 044039. http://doi.org/10.1103/PhysRevD.97.044039
    """

    if loss_function.lower() == 'mae':
        loss = tf.keras.losses.MeanAbsoluteError()
    elif loss_function.lower() == 'mape':
        loss = tf.keras.losses.MeanAbsolutePercentageError()
    else:
        print('Unknown loss function: {}.'.format(loss_function))
        return None

    if optimizer.lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        print('Unknown optimizer: {}.'.format(optimizer))
        return None    

    N_units = {}

    if depth_level == 1:
        N_units['conv'] = [16, 32, 64]
    elif depth_level == 2:
        N_units['conv'] = [64, 128, 256, 512]

    N_units['pooling'] = [4 for _ in range(len(N_units['conv']))]
    kernel_size = [5 for _ in range(len(N_units['conv']))]

    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape((N_samples,1), input_shape=(N_samples,)),
    ])

    for N_conv,N_pooling,sz in zip(N_units['conv'], N_units['pooling'], kernel_size):
        model.add(tf.keras.layers.Conv1D(N_conv, sz, activation=None))
        model.add(tf.keras.layers.MaxPooling1D(N_pooling))
        model.add(tf.keras.layers.ReLU())
    
    model.add(tf.keras.layers.Flatten())
    if depth_level == 2:
        model.add(tf.keras.layers.Dense(128, activation='relu'))
    
    model.add(tf.keras.layers.Dense(64, activation='relu'))

    if dropout_coeff is not None:
        model.add(tf.keras.layers.Dropout(dropout_coeff))

    model.add(tf.keras.layers.Dense(y['training'].shape[1]))

    model.compile(optimizer=optimizer, loss=loss)

    if full_output:
        return model, N_units, kernel_size, optimizer, loss

    return model


def train_network(model, x, y, N_epochs, batch_size, output_dir, steps_per_epoch = None, verbose = 1, full_output = False):
    checkpoint_dir = output_dir + '/checkpoints'
    os.makedirs(checkpoint_dir)

    # create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir + \
                                                     '/weights.{epoch:02d}-{val_loss:.2f}.h5',
                                                     save_weights_only=False,
                                                     save_best_only=True,
                                                     monitor='val_loss',
                                                     verbose=0)

    if steps_per_epoch is None:
        N_batches = np.ceil(x['training'].shape[0] / batch_size)
        steps_per_epoch = np.max([N_batches, 100])

    history = model.fit(x['training'], y['training'], epochs=N_epochs, batch_size=batch_size,
                     steps_per_epoch=steps_per_epoch, validation_data=(x['validation'], y['validation']),
                     verbose=verbose, callbacks=[cp_callback])
    if full_output:
        return history, steps_per_epoch

    return history


if __name__ == '__main__':

    progname = os.path.basename(sys.argv[0])
    
    parser = arg.ArgumentParser(description = 'Train a network to estimate inertia',
                                formatter_class = arg.ArgumentDefaultsHelpFormatter, \
                                prog = progname)
    parser.add_argument('config_file', type=str, action='store', help='configuration file')
    parser.add_argument('-d', '--data-dir', default=None, type=str, help='directory where training data is stored')
    parser.add_argument('-v', '--var-name',  default=None,  type=str, help='name of the variable to use for the training')
    parser.add_argument('-o', '--output-dir',  default='experiments',  type=str, help='output directory')
    parser.add_argument('--no-comet', action='store_true', help='do not use CometML to log the experiment')
    args = parser.parse_args(args=sys.argv[1:])

    config_file = args.config_file
    if not os.path.isfile(config_file):
        print('{}: {}: no such file.'.format(progname, config_file))
        sys.exit(1)
    config = json.load(open(config_file, 'r'))
    
    with open('/dev/random', 'rb') as fid:
        seed = int.from_bytes(fid.read(4), 'little')
    tf.random.set_seed(seed)
    print('Seed: {}'.format(seed))

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

    ### load the data
    if args.data_dir is not None:
        data_folder = args.data_dir
    else:
        data_folder = config['data_dir']
    if not os.path.isdir(data_folder):
        print('{}: {}: no such directory.'.format(progname, data_folder))
        sys.exit(1)

    inertia = {}
    for key in ('training', 'test', 'validation'):
        inertia[key] = np.sort([float(re.findall('[0-9]+\.[0-9]*', f)[-1]) for f in glob.glob(data_folder + '/*' + key + '*.npz')])

    if args.var_name is not None:
        var_name = args.var_name
    else:
        var_name = config['var_name']
    time, x, y = load_data(data_folder, inertia, var_name)
    N_samples = x['training'].shape[1]

    ### normalize the data
    x_train_mean = np.mean(x['training'])
    x_train_std = np.std(x['training'])
    for key in x:
        x[key] = (x[key] - x_train_mean) / x_train_std

    try:
        dropout_coeff = config['dropout']
        if dropout_coeff <= 0:
            dropout_coeff = None
    except:
        dropout_coeff = None

    depth_level = config['depth_level']
    learning_rate = config['learning_rate']
    loss_function =  config['loss_function']
    model, N_units, kernel_size, optimizer, loss = build_network(N_samples, depth_level, learning_rate, dropout_coeff, loss_function, full_output=True)
    model.summary()

    ### train the network
    N_epochs   = config['N_epochs']
    batch_size = config['batch_size']

    parameters = config.copy()
    parameters['seed'] = seed
    parameters['data_dir'] = data_folder
    parameters['var_name'] = var_name
    parameters['dropout_coeff'] = dropout_coeff
    parameters['N_samples'] = N_samples
    parameters['N_units'] = N_units
    parameters['kernel_size'] = kernel_size
    parameters['x_train_mean'] = x_train_mean
    parameters['x_train_std'] = x_train_std
    
    if log_to_comet:
        experiment.log_parameters(parameters)

    output_path = args.output_dir + '/' + experiment_key
    history, steps_per_epoch = train_network(model, x, y, N_epochs, batch_size, \
                                             output_dir = output_path, \
                                             full_output = True)

    parameters['steps_per_epoch'] = steps_per_epoch

    if log_to_comet:
        experiment.log_parameter('steps_per_epoch', steps_per_epoch)
        experiment.set_step(0)

    checkpoint_path = output_path + '/checkpoints'
    
    ### find the best model based on the validation loss
    checkpoint_files = glob.glob(checkpoint_path + '/*.h5')
    val_loss = [float(file[:-3].split('-')[-1]) for file in checkpoint_files]
    best_checkpoint = checkpoint_files[np.argmin(val_loss)]
    best_model = tf.keras.models.load_model(best_checkpoint)

    ### compute the network prediction on the test set
    y_prediction = np.squeeze(best_model.predict(x['test']))

    ### compute the mean absolute percentage error on the CNN prediction
    y_test = np.squeeze(y['test'].numpy())
    mape_prediction = tf.keras.losses.mean_absolute_percentage_error(y_test, y_prediction).numpy()
    print('MAPE on CNN prediction ... {:.2f}%'.format(mape_prediction))
    test_results = {'y_test': y_test, 'y_prediction': y_prediction, 'mape_prediction': mape_prediction}
    
    if log_to_comet:
        experiment.log_metric('mape_prediction', mape_prediction)

    best_model.save(output_path)
    pickle.dump(test_results, open(output_path + '/test_results.pkl', 'wb'))
    pickle.dump(parameters, open(output_path + '/parameters.pkl', 'wb'))
    pickle.dump(history.history, open(output_path + '/history.pkl', 'wb'))

    fig,(ax1,ax2) = plt.subplots(1, 2, figsize=(10,4))
    ### plot the loss as a function of the epoch number
    epochs = np.r_[0 : len(history.history['loss'])] + 1
    ax1.plot(epochs, history.history['loss'], 'k', label='Training')
    ax1.plot(epochs, history.history['val_loss'], 'r', label='Validation')
    ax1.legend(loc='best')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ### plot the results obtained with the CNN
    limits = np.squeeze(y['training'].numpy()[[0,-1]])
    limits[1] += 1
    ax2.plot(limits, limits, 'g--')
    ax2.plot(y['test'], y_prediction, 'o', color=[1,.7,1], markersize=4, \
             markerfacecolor='w', markeredgewidth=1)
    for i in range(int(limits[0]), int(limits[1])):
        idx,_ = np.where(np.abs(y['test'] - (i + 1/3)) < 1e-3)
        m = np.mean(y_prediction[idx])
        s = np.std(y_prediction[idx])
        ax2.plot(i+1/3 + np.zeros(2), m + s * np.array([-1,1]), 'm-', linewidth=2)
        ax2.plot(i+1/3, m, 'ms', markersize=8, markerfacecolor='w', markeredgewidth=2)
    ax2.set_title('CNN')
    ax2.set_xlabel('Expected value')
    ax2.set_ylabel('Predicted value')
    ax2.axis([1.8, limits[1], 0, limits[1]])
    plt.savefig(output_path + '/summary.pdf')

