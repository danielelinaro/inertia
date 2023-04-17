
import os
import sys
import numpy as np
from comet_ml.api import API, APIExperiment
from comet_ml.query import Tag

import colorama as cm
print_error   = lambda msg: print(f'{cm.Fore.RED}'    + msg + f'{cm.Style.RESET_ALL}')
print_warning = lambda msg: print(f'{cm.Fore.YELLOW}' + msg + f'{cm.Style.RESET_ALL}')
print_msg     = lambda msg: print(f'{cm.Fore.GREEN}'  + msg + f'{cm.Style.RESET_ALL}')


__all__ = ['HashKeysDict', 'collect_experiments', 'print_error', 'print_warning', 'print_msg']


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


def collect_experiments(area_IDs, network_name = 'IEEE39', area_measure = 'momentum',
                        stoch_load_bus_IDs=None, rec_bus_IDs=None, additional_tags=None,
                        missing_tags=None, D=None, DZA=None, H_G1=None, time_interval=None,
                        full_metrics=True, verbose_level=1):
    """
    D - damping (default used to be 2)
    DZA - dead-zone amplitude (default used to be 60)
    H_G1 - inertia of generator 1 (default used to be 500)
    rec_bus_IDs - the bus(es) used for recording: an empy list means that the corresponding
                  experiment tag won't be used

    """

    if np.isscalar(area_IDs):
        area_IDs = [area_IDs]

    api = API(api_key = os.environ['COMET_API_KEY'])
    workspace = 'danielelinaro'
    project_name = 'inertia'

    inertia_units = 'GW s'

    query = Tag(network_name) & \
            Tag('area_measure_' + area_measure) & \
            Tag('1D_pipeline') & \
            Tag('_'.join([f'area{ID}' for ID in area_IDs]))

    if stoch_load_bus_IDs is not None and len(stoch_load_bus_IDs) > 0:
        stoch_load_bus_list = 'stoch_load_bus_' + '-'.join(map(str, stoch_load_bus_IDs))
        query &= Tag(stoch_load_bus_list)

    if D is not None:
        query &= Tag(f'D={D}')

    if DZA is not None:
        query &= Tag(f'DZA={DZA}')

    if H_G1 is not None:
        query &= Tag(f'H_G1_{H_G1}')

    if rec_bus_IDs is not None and len(rec_bus_IDs) > 0:
        rec_bus_list = 'buses_' + '-'.join(map(str, rec_bus_IDs))
        query &= Tag(rec_bus_list)

    for tag in additional_tags:
        if isinstance(tag, str):
            query &= Tag(tag)
        else:
            query &= tag

    if verbose_level == 2:
        print('Query:', query)

    if verbose_level >= 1:
        sys.stdout.write('Querying Comet website... ')
        sys.stdout.flush()
    expts = api.query(workspace, project_name, query, archived=False)
    if verbose_level >= 1: sys.stdout.write(f'done: {len(expts)} experiments match the query.\n')

    if time_interval is not None:
        if verbose_level >= 1:
            sys.stdout.write('Getting experiment start times... ')
            sys.stdout.flush()
        # keep only those experiments that are within a certain time interval
        from datetime import datetime
        if not isinstance(time_interval, list) and not isinstance(time_interval, tuple):
            raise Exception('time_interval must be either a list or a tuple')
        window = [datetime(2000, 1, 1).timestamp(), datetime(2100, 1, 1).timestamp()]
        for i in range(2):
            if time_interval[i] is not None:
                window[i] = time_interval[i].timestamp()
        to_keep = []
        for expt in expts:
            start_ts = expt.get_metrics_summary('loss')['timestampMin'] / 1000
            if start_ts >= window[0] and start_ts <= window[1]:
                to_keep.append(expt)
        if verbose_level >= 1:
            if len(to_keep) == len(expts):
                sys.stdout.write(f'done: all experiments started within the requested time interval.\n')
            else:
                sys.stdout.write(f'done: {len(expts)-len(to_keep)} experiments did not start within the requested time interval.\n')
        if len(to_keep) == 0:
            return None
        expts = to_keep

    # the next bit of code keeps only those experiments that do not have tags
    # appearing in the missing_tags list: this is necessary because I can't figure
    # out how to negate a tag directly in a Comet query
    if missing_tags is None:
        missing_tags = ['DO_NOT_USE']
    if isinstance(missing_tags, str):
        missing_tags = [missing_tags]
    if 'DO_NOT_USE' not in missing_tags:
        missing_tags.append('DO_NOT_USE')
    if verbose_level >= 1:
        sys.stdout.write('Getting experiment tags... ')
        sys.stdout.flush()
    experiments = []
    tags = {}
    for expt in expts:
        tags[expt.id] = expt.get_tags()
        keep = True
        for tag in missing_tags:
            if tag in tags[expt.id]:
                keep = False
                break
        if keep:
            experiments.append(expt)
    if verbose_level >= 1: sys.stdout.write(f'done: {len(expts)-len(experiments)} experiments were removed.\n')

    n_experiments = len(experiments)
    if n_experiments == 0:
        return None
    expts = HashKeysDict()
    if verbose_level == 0:
        from tqdm import tqdm
        iter_fun = lambda it: tqdm(it, ascii=True, ncols=70)
    else:
        iter_fun = lambda it: it
    for i in iter_fun(range(n_experiments)):
        expt = experiments[i]
        ID = expt.id
        if verbose_level >= 1:
            sys.stdout.write(f'[{i+1:02d}/{n_experiments:02d}] downloading data for experiment ID {ID}... ')
            sys.stdout.flush()
        expts[ID] = {'tags': tags[ID]}
        for metric_name in ('val_loss','loss', 'batch_loss'):
            if full_metrics:
                expts[ID][metric_name] = np.array([float(m['metricValue']) for m in expt.get_metrics(metric_name)])
            else:
                expts[ID][metric_name] = float(expt.get_metrics_summary(metric_name)['valueMin'])
        metric = expt.get_metrics('mape_prediction')
        if len(metric) > 0:
            val = metric[0]['metricValue']
            try:
                expts[ID]['MAPE'] = float(val)
            except:
                expts[ID]['MAPE'] = np.array(list(map(float, [v for v in val[1:-1].split(' ') if len(v)])))
        else:
            expts[ID]['MAPE'] = np.nan    
        if verbose_level >= 1: sys.stdout.write('done.\n')
        if verbose_level == 2:
            print('  val_loss: {:.4f}'.format(expts[ID]['val_loss'].min()))
            if expts[ID]['MAPE'] is not None:
                try:
                    print('      MAPE: {:.3f}%'.format(expts[ID]['MAPE']))
                except:
                    print('      MAPE: {}%'.format('%, '.join([f'{m:.3f}' for m in expts[ID]['MAPE']])))
            else:
                print('      MAPE: [experiment not terminated]')
            print('      Tags: "{}"'.format('" "'.join(expts[ID]['tags'])))
        if expts[ID]['MAPE'] is None:
            expts.pop(ID)
    return expts


