
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


def collect_experiments(area_IDs, network_name = 'IEEE39',
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

    if len(rec_bus_IDs) > 0:
        rec_bus_list = 'buses_' + '-'.join(map(str, rec_bus_IDs))
        query &= Tag(rec_bus_list)

    for tag in additional_tags:
        if isinstance(tag, str):
            query &= Tag(tag)
        else:
            query &= tag

    if verbose:
        print('Query:', query)
    experiments = api.query(workspace, project_name, query, archived=False)
    n_experiments = len(experiments)
    if n_experiments == 0:
        return None
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


