import os
import sys
import warnings
import torch
import pickle
import numpy as np

from itertools import product


HOST_NAME = os.popen('hostname').read()
if 'della' in HOST_NAME:
    LOG_ROOT = '/tigress/qlu/logs/free-recall/log'
    FIG_ROOT = '/tigress/qlu/logs/free-recall/figs'
else:
    LOG_ROOT = '../log'
    FIG_ROOT = '../figs'

print(f'Host  = {HOST_NAME}')
print(f'LOG_ROOT = {LOG_ROOT}')
print(f'FIG_ROOT = {FIG_ROOT}')

eps = np.finfo(np.float32).eps.item()


def to_pth(np_array, pth_dtype=torch.FloatTensor):
    return torch.tensor(np_array).type(pth_dtype)


def to_sqpth(np_array, pth_dtype=torch.FloatTensor):
    return torch.squeeze(to_pth(np_array, pth_dtype=pth_dtype))


def to_np(torch_tensor):
    return torch_tensor.data.numpy()


def to_sqnp(torch_tensor, dtype=np.float):
    return np.array(np.squeeze(to_np(torch_tensor)), dtype=dtype)


def enumerated_product(*args):
    # https://stackoverflow.com/questions/56430745/enumerating-a-tuple-of-indices-with-itertools-product
    yield from zip(product(*(range(len(x)) for x in args)), product(*args))

def estimated_run_time(time_took_per_epoch, n_epochs):
    return time_took_per_epoch * n_epochs / 3600

def rm_dup(seq):
    # https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def int2onehot(num, dim):
    return torch.eye(dim)[num]

def make_log_fig_dir(exp_name, log_root=LOG_ROOT, fig_root=FIG_ROOT, makedirs=True):
    log_path = os.path.join(log_root, exp_name)
    if not os.path.exists(log_path):
        if makedirs:
            os.makedirs(log_path)
            print(f'made log dir: {log_path}')
        else:
            print(f'log dir not found: {log_path}')

    fig_path = os.path.join(fig_root, exp_name)
    if not os.path.exists(fig_path) and makedirs:
        if makedirs:
            os.makedirs(fig_path)
            print(f'made fig dir: {fig_path}')
        else:
            print(f'fig dir not found: {log_path}')
    return log_path, fig_path

def ignore_warnings():
    if not sys.warnoptions:
        warnings.simplefilter("ignore")


def pickle_save_dict(input_dict, save_path):
    """Save the dictionary
    Parameters
    ----------
    input_dict : type
        Description of parameter `input_dict`.
    save_path : type
        Description of parameter `save_path`.
    """
    with open(save_path, 'wb') as handle:
        pickle.dump(input_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load_dict(fpath):
    """load the dict
    Parameters
    ----------
    fpath : type
        Description of parameter `fpath`.
    Returns
    -------
    type
        Description of returned object.
    """
    return pickle.load(open(fpath, "rb"))
