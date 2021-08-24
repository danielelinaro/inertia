
import os
import re
import sys
import tables
import numpy as np
from comet_ml.api import API, APIExperiment
from comet_ml.query import Tag

__all__ = ['HashKeysDict', 'read_area_values', 'load_one_block', 'load_data_areas',
           'load_data_generators', 'load_data_slide', 'predict', 'slide_window',
           'collect_experiments']

default_H = {
    'IEEE14': {
        1: 10.296 / 2,
        2: 13.08 / 2,
        3: 13.08 / 2,
        6: 10.12 / 2,
        8: 10.12 / 2
    },
    'two-area': {
        1: 6.5,
        2: 6.5,
        3: 6.175,
        4: 6.175
    }
}


class HashKeysDict (dict):
    def __init__(self, *args):
        dict.__init__(self, args)

    def find_key(self, key):
        if key in self:
            return key
        for k in self:
            if k.startswith(key):
                return k
        return None

    def __getitem__(self, key):
        return dict.__getitem__(self, self.find_key(key))

    def __setitem__(self, key, val):
        k = self.find_key(key)
        if k is None:
            k = key
        dict.__setitem__(self, k, val)


def read_area_values(filename, generators_areas_map = None, generators_Pnom = None, area_measure = 'inertia'):
    fid = tables.open_file(filename, 'r')
    pars = fid.root.parameters.read()
    fid.close()
    generator_IDs = [gen_ID.decode('utf-8') for gen_ID in pars['generator_IDs'][0]]
    generator_inertias = pars['inertia'][0]
    if generators_areas_map is not None:
        N_areas = len(generators_areas_map)
        area_measures = np.zeros(N_areas)
        for i,area_generators in enumerate(generators_areas_map):
            num, den = 0, 0
            for gen_ID in area_generators:
                idx = generator_IDs.index(gen_ID)
                num += generator_inertias[idx] * generators_Pnom[gen_ID]
                den += generators_Pnom[gen_ID]
                if area_measure.lower() == 'inertia':
                    area_measures[i] = num / den   # [s]
                elif area_measure.lower() == 'energy':
                    area_measures[i] = num * 1e-9  # [GW s]
                elif area_measure.lower() == 'momentum':
                    area_measures[i] = 2 * num * 1e-9 / 60.  # [GW s^2]
        return generator_IDs, generator_inertias, area_measures
    return generator_IDs, generator_inertias


def load_one_block(filename, var_names, trial_dur = 60, max_num_rows = np.inf, dtype = np.float32, add_omega_ref = True):
    """
    This function loads a single data file containing the results of the simulations for one value of inertia
    """
    ext = os.path.splitext(filename)[1]

    fid = tables.open_file(filename, 'r')
    # do not convert time to dtype here because that gives problems when computing n_samples below
    time = fid.root.time.read()
    if len(fid.root[var_names[0]].shape) > 1:
        X = [fid.root[var_name].read(stop=np.min([fid.root[var_name].shape[0], max_num_rows])) for var_name in var_names]
    else:
        X = [fid.root[var_name].read() for var_name in var_names]
    if 'omega_ref' in fid.root and add_omega_ref:
        if len(fid.root[var_names[0]].shape) > 1:
            omega_ref = fid.root.omega_ref.read(stop=np.min([fid.root.omega_ref.shape[0], max_num_rows])) - 1
        else:
            omega_ref = fid.root.omega_ref.read() - 1
        for i,var_name in enumerate(var_names):
            if var_name != 'omega_ref' and 'omega' in var_name:
                X[i] += omega_ref
    pars = fid.root.parameters.read()
    inertia = pars['inertia'][0]
    generator_IDs = [gen_ID.decode('utf-8') for gen_ID in pars['generator_IDs'][0]]
    fid.close()

    dt = np.diff(time[:2])[0]
    orig_n_trials, orig_n_samples = X[0].shape
    n_samples = int(trial_dur / dt)
    n_trials = int(orig_n_trials * orig_n_samples / n_samples)
    time = time[:n_samples]
    stop = orig_n_samples % n_samples
    X = np.array([np.reshape(x[:,:orig_n_samples-stop], [n_trials, n_samples], order='C') for x in X], dtype=dtype)

    return time.astype(dtype), X, inertia, generator_IDs


def load_data_areas(data_files, var_names, generators_areas_map, generators_Pnom, area_measure,
                    max_block_size = np.inf, dtype = np.float32, use_tf = True, add_omega_ref = True):
    """
    area_measure - whether Y should contain the inertia of the coi or the total energy of the area
    """
    # Note: dtype should be a NumPy type, even if use_tf = True
    X = {}
    Y = {}

    if area_measure.lower() not in ('inertia', 'energy', 'momentum'):
        raise Exception('area_measure must be one of "inertia", "energy" or "momentum"')

    n_areas = len(generators_areas_map)
    for key in data_files:
        for data_file in data_files[key]:
            time, x, h, generator_IDs = load_one_block(data_file, var_names, 60, max_block_size, dtype, add_omega_ref)
            y = np.zeros(n_areas, dtype=dtype)
            for i,area_generators in enumerate(generators_areas_map):
                num = 0
                den = 0
                for gen_ID in area_generators:
                    idx = generator_IDs.index(gen_ID)
                    num += h[idx] * generators_Pnom[gen_ID]
                    den += generators_Pnom[gen_ID]
                if area_measure.lower() == 'inertia':
                    y[i] = num / den          # [s]
                elif area_measure.lower() == 'energy':
                    y[i] = num * 1e-9         # [GW s]
                elif area_measure.lower() == 'momentum':
                    y[i] = 2 * num * 1e-9 / 60.  # [GW s^2]
            y = np.tile(y, [x.shape[1], 1])
            try:
                X[key] = np.concatenate((X[key], x), axis=1)
                Y[key] = np.concatenate((Y[key], y), axis=0)
            except:
                X[key] = x
                Y[key] = y

    if use_tf:
        import tensorflow as tf
        time = tf.constant(time, dtype=tf.dtypes.as_dtype(dtype))
        for key in X:
            X[key] = tf.constant(X[key], dtype=tf.dtypes.as_dtype(dtype))
            Y[key] = tf.constant(Y[key], dtype=tf.dtypes.as_dtype(dtype))
    return time, X, Y


def load_data_generators(folders, generator_IDs, inertia_values, var_names,
                         max_block_size = np.inf, dtype = np.float32, use_tf = True):
    # Note: dtype should be a NumPy type, even if use_tf = True

    X = {}
    Y = {}
    if isinstance(inertia_values, dict):
        inertias = [inertia_values for _ in folders]
    else:
        inertias = inertia_values
    n_outputs = len(generator_IDs)
    for i, (folder, inertia) in enumerate(zip(folders, inertias)):
        if folder[-1] != '/':
            folder += '/'
        for key,H in inertia.items():
            for h in H:
                filename = folder + 'H_{:.3f}_{}_set.h5'.format(h, key)
                time, x, _ = load_one_block(filename, var_names, 60, max_block_size, dtype)
                y = np.zeros((x.shape[-2], n_outputs), dtype=dtype)
                y[:,i] = h
                for j in range(n_outputs):
                    if j != i:
                        y[:,j] = default_H['IEEE14'][generator_IDs[j]]
                try:
                    X[key] = np.concatenate((X[key], x), axis=1)
                    Y[key] = np.concatenate((Y[key], y), axis=0)
                except:
                    X[key] = x
                    Y[key] = y
    if use_tf:
        import tensorflow as tf
        time = tf.constant(time, dtype=tf.dtypes.as_dtype(dtype))
        for key in X:
            X[key] = tf.constant(X[key], dtype=tf.dtypes.as_dtype(dtype))
            Y[key] = tf.constant(Y[key], dtype=tf.dtypes.as_dtype(dtype))
    return time, X, Y

            
def slide_window(X, window_size, overlap=None, window_step_size=None, N_windows=-1):
    # window_size and window_step_size are in units of samples
    if overlap is not None and window_step_size is not None:
        raise 'Only one of "overlap" and "window_step_size" should be passed'
    if overlap is None and window_step_size is None:
        raise 'One of "overlap" and "window_step_size" must be passed'
    if window_step_size is None:
        window_step_size = int(window_size * overlap)
    if N_windows <= 0:
        N_windows = (X.size - window_size) // window_step_size
    idx = np.zeros((N_windows, window_size), dtype=int)
    Y = np.zeros((N_windows, window_size))
    for i in range(N_windows):
        idx[i,:] = i * window_step_size + np.r_[0 : window_size]
        try:
            Y[i,:] = X[idx[i,:]]
        except:
            print('>>> {:04d}/{:04d}'.format(i, N_windows))
            break
    return Y, idx

def load_data_slide(data_files, var_names, data_mean = None, data_std = None, window_dur = 60, window_step = 1,
                    ttran = 0, normalize_sliding=False, add_omega_ref=True, verbose=False):
    # window_dur and window_step are in units of seconds
    fids = [tables.open_file(data_file, 'r') for data_file in data_files]
    n_files = len(data_files)
    params = fids[0].root.parameters.read()
    dt = 1 / params['frand'][0]
    window_size = int(window_dur / dt)
    window_step_size = int(window_step / dt)
    if verbose:
        print(f'Window size: {window_size} samples')
        print(f'Window step size: {window_step_size} samples')
    time = [fids[0].root.time.read()]
    for i in range(1, n_files):
        time.append(fids[i].root.time.read() + time[i-1][-1])
    time = np.concatenate(time)
    idx = time > ttran
    N_samples = time.size
    data = {var_name: np.concatenate([fid.root[var_name].read() for fid in fids]) for var_name in var_names}
    if add_omega_ref:
        try:
            omega_ref = np.concatenate([fid.root.omega_ref.read() for fid in fids]) - 1
            for var_name in var_names:
                if var_name != 'omega_ref' and 'omega' in var_name:
                    data[var_name] += omega_ref
        except:
            pass
    if not normalize_sliding:
        if data_mean is None:
            data_mean = {var_name: np.mean(data[var_name][idx]) for var_name in var_names}
            print(f'data_mean = {data_mean}')
        if data_std is None:
            data_std = {var_name: np.std(data[var_name][idx]) for var_name in var_names}
            print(f'data_std = {data_std}')
        data_normalized = {var_name: (data[var_name] - data_mean[var_name]) / data_std[var_name] for var_name in var_names}
        data_to_split = data_normalized
    else:
        data_normalized = None
        data_to_split = data
    data_sliding = {}
    indexes = {}
    for var_name in var_names:
        data_sliding[var_name], indexes[var_name] = slide_window(data_to_split[var_name],
                                                                 window_size,
                                                                 window_step_size=window_step_size)
        if normalize_sliding:
            if data_mean is None:
                mu = np.tile(data_sliding[var_name].mean(axis=1), [data_sliding[var_name].shape[1], 1]).T
            else:
                mu = data_mean[var_name]
            if data_std is None:
                sigma = np.tile(data_sliding[var_name].std(axis=1), [data_sliding[var_name].shape[1], 1]).T
            else:
                sigma = data_std[var_name]
            data_sliding[var_name] = (data_sliding[var_name] - mu) / sigma

    if verbose:
        print('Number of trials: {:d}'.format(data_sliding[var_names[0]].shape[0]))

    for fid in fids:
        fid.close()
    
    return time, data, data_normalized, data_sliding, indexes


def predict(model, data_sliding, window_step, rolling_length=50):
    # window_step is in seconds
    import tensorflow as tf
    import pandas as pd
    var_names = data_sliding.keys()
    if len(model.inputs) > 1:
        x = {var_name: tf.constant(data_sliding[var_name], dtype=tf.float32) for var_name in var_names}
    elif len(model.inputs) == 1 and len(data_sliding.keys()) > 1:
        x = tf.constant(list(data_sliding.values()), dtype=tf.float32)
        x = tf.transpose(x, perm=(1,2,0))
    else:
        x = tf.constant(data_sliding[var_names[0]], dtype=tf.float32)
    y = model.predict(x)
    n_samples, n_outputs = y.shape
    data = {f'inertia_{i}': y[:,i] for i in range(n_outputs)}
    H = pd.DataFrame(data).rolling(rolling_length).mean().to_numpy()
    time = np.arange(n_samples) * window_step
    return time, H, y

def collect_experiments(area_ID, network_name = 'IEEE39',
                        area_measure = 'inertia',
                        D=2, DZA=60, H_G1=500,
                        stoch_load_bus_IDs = [3],
                        rec_bus_IDs = [],
                        additional_tags = [],
                        verbose = False):
    """
    D - damping
    DZA - dead-zone amplitude
    H_G1 - inertia of generator 1
    rec_bus_IDs - the bus(es) used for recording: an empy list means that the corresponding
                  experiment tag won't be used

    """
    api = API(api_key = os.environ['COMET_API_KEY'])
    workspace = 'danielelinaro'
    project_name = 'inertia'

    inertia_units = 'GW s'
    stoch_load_bus_list = 'stoch_load_bus_' + '-'.join(map(str, stoch_load_bus_IDs))

    query = Tag(network_name) & \
            Tag('area_measure_' + area_measure) & \
            Tag(f'D={D}') & \
            Tag(f'DZA={DZA}') & \
            Tag('1D_pipeline') & \
            Tag(stoch_load_bus_list) & \
            Tag(f'H_G1_{H_G1}') & \
            Tag(f'area{area_ID}')

    if len(rec_bus_IDs) > 1:
        rec_bus_list = 'buses_' + '-'.join(map(str, rec_bus_IDs))
        query &= Tag(rec_bus_list)

    for tag in additional_tags:
        if isinstance(tag, str):
            query &= Tag(tag)
        else:
            query &= tag

    experiments = api.query(workspace, project_name, query, archived=False)
    n_experiments = len(experiments)
    expts = HashKeysDict()
    for i,experiment in enumerate(experiments):
        ID = experiment.id
        sys.stdout.write(f'[{i+1:02d}/{n_experiments:02d}] downloading data for experiment ID {ID}... ')
        metrics = experiment.get_metrics()
        sys.stdout.write('done.\n')
        val_loss = []
        loss = []
        batch_loss = []
        mape = None
        for m in metrics:
            if m['metricName'] == 'val_loss':
                val_loss.append(float(m['metricValue']))
            elif m['metricName'] == 'loss':
                loss.append(float(m['metricValue']))
            elif m['metricName'] == 'batch_loss':
                batch_loss.append(float(m['metricValue']))
            elif m['metricName'] == 'mape_prediction':
                val = m['metricValue']
                try:
                    mape = float(val)
                except:
                    mape = np.array(list(map(float, [v for v in val[1:-1].split(' ') if len(v)])))
        expts[ID] = {
            'loss': np.array(loss),
            'val_loss': np.array(val_loss),
            'batch_loss': np.array(batch_loss),
            'MAPE': mape,
            'tags': experiment.get_tags()
        }
        if verbose:
            print('  val_loss: {:.4f}'.format(expts[ID]['val_loss'].min()))
            if expts[ID]['MAPE'] is not None:
                print('      MAPE: {:.4f}%'.format(expts[ID]['MAPE']))
            else:
                print('      MAPE: [experiment not terminated]')
            print('      Tags: "{}"'.format('" "'.join(expts[ID]['tags'])))
        if expts[ID]['MAPE'] is None:
            expts.pop(ID)
    return expts
