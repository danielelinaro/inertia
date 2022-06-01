
import os
import sys
import glob
import pickle
from scipy.signal import butter, filtfilt, hilbert

import numpy as np

import tensorflow as tf
from tensorflow import keras
from comet_ml.api import API, APIExperiment

from dlml.utils import collect_experiments
from dlml.data import load_data_areas
from dlml.nn import compute_receptive_field, compute_correlations

prog_name = os.path.basename(sys.argv[0])

def usage():
    print(f'usage: {prog_name} [<options>] <experiment_ID>')
    print('')
    print('    -N, --nbands   number of frequency bands in which the range (0.05,Fn)')
    print('                    will be subdivided (default 20, with Fn the Nyquist frequency)')
    print('    --stop-layer   name of the layer used for the computation of correlations')
    print('    -o, --output   output file name')
    print('    -f, --force    force overwrite of existing data file')
    print('    --plots        generate plots')
    print('    -h, --help     print this help message and exit')
    print('')
    print(f'Run with {prog_name} 034a1edb0797475b985f0e1335dab383')


def plot_correlations(R, p, R_ctrl, p_ctrl, edges, F, Xf, idx, sort_F=1.0, vmin=None, vmax=None):
    import matplotlib.pyplot as plt
    import matplotlib
    fntsize = 8
    matplotlib.rc('font', size=fntsize)

    if p is not None:
        R = R.copy()
        R[p > 0.05] = 0
    if p_ctrl is not None:
        R_ctrl = R_ctrl.copy()
        R_ctrl[p_ctrl > 0.05] = 0
    R_mean = [R[jdx].mean(axis=0) for jdx in idx]
    R_ctrl_mean = [R_ctrl[jdx].mean(axis=0) for jdx in idx]
    rows = len(idx)    
    edge = np.abs(edges - sort_F).argmin()
    for i in range(rows):
        kdx = np.argsort(R_mean[i][edge,:])
        R_mean[i] = R_mean[i][:,kdx]
        kdx = np.argsort(R_ctrl_mean[i][edge,:])
        R_ctrl_mean[i] = R_ctrl_mean[i][:,kdx]

    fig = plt.figure(figsize=(8, 3*rows))
    offset = 0.02, 0.01 + max([0.08 - rows * 0.01, 0])
    border = 0.05, 0.01 + max([0.06 - rows * 0.01, 0])
    space = 0.1, 0.025
    w = 0.17, 0.29, 0.325
    h = 0.75
    h = (1 - offset[1] - border[1] - space[1]*(rows-1)) / rows
    cols = 3
    ax = []
    for i in range(rows):
        y = 1 - border[1] - (i+1) * h - i * space[1]
        ax.append([fig.add_axes([offset[0], y, w[0], h]),
                   fig.add_axes([offset[0] + w[0] + space[0], y, w[1], h]),
                   fig.add_axes([offset[0] + np.sum(w[:2]) + np.sum(space), y, w[2], h])])
    cmap = plt.get_cmap('tab10', len(idx))
    fft_max = 0
    for j,jdx in enumerate(idx):
        mean = Xf[jdx].mean(axis=0)
        stddev = Xf[jdx].std(axis=0)
        ci = 1.96 * stddev / np.sqrt(jdx.size)
        m = np.max(mean[F > 0.1] + ci[F > 0.1])
        if m > fft_max:
            fft_max = m
        for i in range(rows):
            ax[i][0].fill_betweenx(F, mean + ci, mean - ci, color=cmap(j))
    for i in range(rows):
        ax[i][0].grid(which='both', axis='y', ls=':', lw=0.5, color=[.6,.6,.6])
        ax[i][0].set_xlim([0, m*1.05])
        ax[i][0].invert_xaxis()
        ax[i][0].yaxis.tick_right()
        for side in 'left','top':
            ax[i][0].spines[side].set_visible(False)

    make_symmetric = False
    if vmin is None:
        vmin = min([r.min() for r in R_mean])
        make_symmetric = True
    if vmax is None:
        vmax = max([r.max() for r in R_mean])
        if make_symmetric:
            if vmax > np.abs(vmin):
                vmin = -vmax
            else:
                vmax = -vmin
    print(f'Color bar bounds: ({vmin:.2f},{vmax:.2f}).')
    ticks = np.linspace(vmin, vmax, 7)
    ticklabels = [f'{tick:.2f}' for tick in ticks]

    cmap = plt.get_cmap('bwr')
    y = edges[:-1] + np.diff(edges) / 2
    for i in range(rows):
        for j,R in enumerate((R_mean[i], R_ctrl_mean[i])):
            x = np.arange(R.shape[-1])
            im = ax[i][j+1].pcolormesh(x, y, R, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)
            for side in 'right','top':
                ax[i][j+1].spines[side].set_visible(False)
            ax[i][j+1].set_xticks(np.linspace(0, x[-1], 3, dtype=np.int32))
        cbar = plt.colorbar(im, fraction=0.1, shrink=1, aspect=20, label='Correlation',
                            orientation='vertical', ax=ax[i][-1], ticks=ticks)
        cbar.ax.set_yticklabels(ticklabels, fontsize=fntsize-1)

    for i in range(rows):
        for j in range(cols):
            ax[i][j].set_ylim(edges[[0,-2]])
            ax[i][j].set_yscale('log')
            if j in (0,2):
                ax[i][j].set_yticklabels([])
            if i < rows-1:
                ax[i][j].set_xticklabels([])
            if j > 0:
                ax[-1][j].set_xlabel('Filter #')
        ax[i][1].set_ylabel('Frequency [Hz]')
    ax[0][1].set_title('Trained network', fontsize=fntsize+1)
    ax[0][2].set_title('Untrained network', fontsize=fntsize+1)

    return fig, vmin, vmax


if __name__ == '__main__':
    
    pooling_type = ''
    if len(sys.argv) == 1 or sys.argv[1] in ('-h','--help'):
        usage()
        sys.exit(0)

    i = 1
    n_args = len(sys.argv)

    output_file = None
    make_plots = False
    force = False
    N_bands = 20
    filter_order = 8
    spacing = 'log'
    stop_layer = None

    while i < n_args:
        arg = sys.argv[i]
        if arg in ('-o', '--output'):
            output_file = sys.argv[i+1]
            i += 1
        elif arg in ('-f', '--force'):
            force = True
        elif arg == '--plots':
            make_plots = True
        elif arg in ('-N', '--nbands'):
            N_bands = int(sys.argv[i+1])
            i += 1
        elif arg == '--spacing':
            if sys.argv[i+1].lower() in ('lin', 'linear'):
                spacing = 'lin'
            elif sys.argv[i+1].lower() in ('log', 'logarithmic'):
                spacing = 'log'
            else:
                raise Exception(f'Unknown value for --spacing: "{sys.argv[i+1]}"')
            i += 1
        elif arg == '--order':
            filter_order = int(sys.argv[i+1])
            i += 1
        elif arg == '--stop-layer':
            stop_layer = sys.argv[i+1]
            i += 1
        else:
            break
        i += 1

    if i == n_args:
        usage()
        sys.exit(1)

    if output_file is not None and os.path.isfile(output_file) and not force and not make_plots:
        print(f'{output_file} exists: use -f to overwrite')
        sys.exit(2)

    if output_file is not None:
        output_file = os.path.splitext(output_file)[0]

    ### Make sure that we have the model requested by the user
    experiment_ID = sys.argv[i]
    experiments_path = 'experiments/neural_network'
    model_dir = os.path.join(experiments_path, experiment_ID)
    if not os.path.isdir(model_dir):
        print(f'{prog_name}: {model_dir}: no such directory')
        sys.exit(3)

    network_parameters = pickle.load(open(os.path.join(model_dir, 'parameters.pkl'), 'rb'))
    if 'use_fft' in network_parameters and network_parameters['use_fft']:
        raise Exception('This script assumes that the input data be in the time domain')

    ### Get some info about the model
    api = API(api_key = os.environ['COMET_API_KEY'])
    experiment = api.get_experiment('danielelinaro', 'inertia', experiment_ID)
    sys.stdout.write(f'Getting metrics for experiment {experiment_ID[:6]}... ')
    sys.stdout.flush()
    metrics = experiment.get_metrics()
    sys.stdout.write('done.\n')
    val_loss = []
    for m in metrics:
        if m['metricName'] == 'val_loss':
            val_loss.append(float(m['metricValue']))
        elif m['metricName'] == 'mape_prediction':
            MAPE = float(m['metricValue'])
    val_loss = np.array(val_loss)

    ### Load the model
    try:
        pooling_type = network_parameters['model_arch']['pooling_type']
    except:
        pooling_type = ''
    checkpoint_path = os.path.join(model_dir, 'checkpoints')
    checkpoint_files = glob.glob(checkpoint_path + '/*.h5')
    try:
        epochs = [int(os.path.split(file)[-1].split('.')[1].split('-')[0]) for file in checkpoint_files]
        best_checkpoint = checkpoint_files[epochs.index(np.argmin(val_loss) + 1)]
    except:
        best_checkpoint = checkpoint_files[-1]
    try:
        model = keras.models.load_model(best_checkpoint)
        custom_objects = None
    except:
        if pooling_type == 'downsample':
            from dlml.nn import DownSampling1D
            custom_objects = {'DownSampling1D': DownSampling1D}
        elif pooling_type == 'spectral':
            from dlml.nn import SpectralPooling
            custom_objects = {'SpectralPooling': SpectralPooling}
        elif pooling_type == 'argmax':
            from dlml.nn import MaxPooling1DWithArgmax
            custom_objects = {'MaxPooling1DWithArgmax': MaxPooling1DWithArgmax}
        with keras.utils.custom_object_scope(custom_objects):
            model = keras.models.load_model(best_checkpoint)
    
    if pooling_type == 'argmax':
        for layer in model.layers:
            if isinstance(layer, MaxPooling1DWithArgmax):
                print(f'Setting store_argmax = True for layer "{layer.name}".')
                layer.store_argmax = True
    x_train_mean = network_parameters['x_train_mean']
    x_train_std  = network_parameters['x_train_std']
    var_names = network_parameters['var_names']
    print(f'Loaded network from {best_checkpoint}.')
    print(f'Variable names: {var_names}')

    model.summary()

    ### Compute effective receptive field size and stride
    if stop_layer is None:
        effective_RF_size,effective_stride = compute_receptive_field(model, stop_layer=keras.layers.Flatten,
                                                                     include_stop_layer=False)
    else:
        effective_RF_size,effective_stride = compute_receptive_field(model, stop_layer=stop_layer,
                                                                     include_stop_layer=True)
    print('Effective receptive field size:')
    for i,(k,v) in enumerate(effective_RF_size.items()):
        print(f'{i}. {k} ' + '.' * (20 - len(k)) + ' {:d}'.format(v))
    print()
    print('Effective stride:')
    for i,(k,v) in enumerate(effective_stride.items()):
        print(f'{i}. {k} ' + '.' * (20 - len(k)) + ' {:d}'.format(v))

    ### Load the data set
    set_name = 'test'
    data_dirs = []
    for area_ID,data_dir in zip(network_parameters['area_IDs'], network_parameters['data_dirs']):
        data_dirs.append(data_dir.format(area_ID))
    data_dir = data_dirs[0]
    data_files = sorted(glob.glob(data_dir + os.path.sep + f'*_{set_name}_set.h5'))
    ret = load_data_areas({set_name: data_files}, network_parameters['var_names'],
                          network_parameters['generators_areas_map'][:1],
                          network_parameters['generators_Pnom'],
                          network_parameters['area_measure'],
                          trial_dur=network_parameters['trial_duration'],
                          max_block_size=100,
                          use_tf=False, add_omega_ref=True,
                          use_fft=False)
    
    t = ret[0]
    X = [(ret[1][set_name][i] - m) / s for i,(m,s) in enumerate(zip(x_train_mean, x_train_std))]
    y = ret[2][set_name]

    ### Predict the momentum using the model
    IDX = [np.where(y == mom)[0] for mom in np.unique(y)]
    n_mom_values = len(IDX)
    momentum = [np.squeeze(model.predict(X[0][jdx])) for jdx in IDX]
    mean_momentum = [m.mean() for m in momentum]
    stddev_momentum = [m.std() for m in momentum]
    print('Mean momentum (with optimized weights):', mean_momentum)
    print(' Std momentum (with optimized weights):', stddev_momentum)

    ### Clone the trained model
    # This initializes the cloned model with new random weights and will be used in the
    # following as a control for the correlation analysis
    #reinit_model = keras.models.clone_model(model)
    reinit_model = model.__class__.from_config(model.get_config(), custom_objects)
    if custom_objects is not None:
        # we have some subclassed layers
        for i in range(len(model.layers)):
            reinit_model.layers[i]._name = model.layers[i].name
    reinit_momentum = [np.squeeze(reinit_model.predict(X[0][jdx])) for jdx in IDX]
    mean_reinit_momentum = [m.mean() for m in reinit_momentum]
    stddev_reinit_momentum = [m.std() for m in reinit_momentum]
    print('Mean momentum (with random weights):', mean_reinit_momentum)
    print(' Std momentum (with random weights):', stddev_reinit_momentum)

    ### Build a model with as many outputs as there are convolutional or pooling layers
    outputs = [layer.output for layer in model.layers \
               if layer.name in effective_RF_size.keys() and not isinstance(layer, keras.layers.InputLayer)]
    multi_output_model = keras.Model(inputs=model.inputs, outputs=outputs)
    # build a control model with the same (multiple-output) architecture as the previous one but random weights:
    ctrl_outputs = [layer.output for layer in reinit_model.layers \
                    if layer.name in effective_RF_size.keys() and not isinstance(layer, keras.layers.InputLayer)]
    ctrl_model = keras.Model(inputs=reinit_model.inputs, outputs=ctrl_outputs)
    print(f'The model has {len(outputs)} outputs, corresponding to the following layers:')
    for i,layer in enumerate(multi_output_model.layers):
        if not isinstance(layer, keras.layers.InputLayer):
            print(f'    {i}. {layer.name}')

    ### Correlations in the actual model
    # define some variables used here and for the control model below:
    dt = np.diff(t[:2])[0]
    fs = np.round(1/dt)
    if spacing == 'lin':
        edges = np.linspace(0.05, 0.5/dt, N_bands+1)
    else:
        edges = np.logspace(np.log10(0.05), np.log10(0.5/dt), N_bands+1)
    edges_ctrl = edges
    bands = [[a,b] for a,b in zip(edges[:-1], edges[1:])]
    N_bands = len(bands)
    _, N_neurons, N_filters = multi_output_model.layers[-1].output.shape
    N_trials = X[0].shape[0]
    if output_file is None:
        output_file = os.path.join(model_dir,
                                   f'correlations_{experiment_ID[:6]}_{N_bands}-bands_' + \
                                   f'{N_filters}-filters_{N_neurons}-neurons_{N_trials}-trials_' + \
                                   f'{filter_order}-butter_{multi_output_model.layers[-1].name}')

    if os.path.isfile(output_file + '.npz') and not force and not make_plots:
        print(f'Output file {output_file}.npz exists: re-run with -f if you want to overwrite it')
        sys.exit(3)

    if not os.path.isfile(output_file + '.npz') or force:
        # compute the correlations:
        R,p = compute_correlations(multi_output_model, X[0], fs, bands, effective_RF_size,
                                   effective_stride, filter_order)
        # compute the correlations for the control model:
        R_ctrl,p_ctrl = compute_correlations(ctrl_model, X[0], fs, bands, effective_RF_size,
                                             effective_stride, filter_order)
        # save everyting
        np.savez_compressed(output_file + '.npz', R=R, p=p, R_ctrl=R_ctrl, p_ctrl=p_ctrl,
                            edges=edges, momentum=y.squeeze(), idx=IDX)
    else:
        data = np.load(output_file + '.npz')
        R, p = data['R'], data['p']
        R_ctrl, p_ctrl = data['R_ctrl'], data['p_ctrl']

    ### Plot the results
    if make_plots:
        data_files_training = sorted(glob.glob(data_dir + os.path.sep + f'*_training_set.h5'))
        if len(data_files_training) == 0:
            data_files_training = sorted(glob.glob(data_dir + os.path.sep + f'*_test_set.h5'))
        ret_fft = load_data_areas({'training': data_files_training}, network_parameters['var_names'],
                                  network_parameters['generators_areas_map'][:1],
                                  network_parameters['generators_Pnom'],
                                  network_parameters['area_measure'],
                                  trial_dur=network_parameters['trial_duration'],
                                  max_block_size=200,
                                  use_tf=False, add_omega_ref=True,
                                  use_fft=True)
        x_train_min_fft = np.array([val.min() for val in ret_fft[1]['training']], dtype=np.float32)
        x_train_max_fft = np.array([val.max() for val in ret_fft[1]['training']], dtype=np.float32)
        ret = load_data_areas({set_name: data_files}, network_parameters['var_names'],
                              network_parameters['generators_areas_map'][:1],
                              network_parameters['generators_Pnom'],
                              network_parameters['area_measure'],
                              trial_dur=network_parameters['trial_duration'],
                              max_block_size=100,
                              use_tf=False, add_omega_ref=True,
                              use_fft=True, Wn=0) # do not filter the data before computing the FFT
        F = ret[0]
        Xf = [(ret[1][set_name][i] - m) / (M - m) for i,(m,M) in enumerate(zip(x_train_min_fft,
                                                                               x_train_max_fft))]
        sort_F = 1.1
        fig,_,_ = plot_correlations(R, p, R_ctrl, p_ctrl, edges, F, Xf[0], IDX, sort_F=sort_F)
        fig.savefig(output_file + '.pdf')

