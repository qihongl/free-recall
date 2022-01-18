import os
import sys
import warnings
import torch
import pickle
import numpy as np

from itertools import product
# from scipy.stats import sem
# from torch.nn.functional import smooth_l1_loss
# from copy import deepcopy


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

#
# def ignore_warnings():
#     if not sys.warnoptions:
#         warnings.simplefilter("ignore")
#
#
# def to_1d_tensor(scalar_list):
#     return torch.cat(
#         [s.type(torch.FloatTensor).view(tensor_length(s)) for s in scalar_list]
#     )
#
#
# def tensor_length(tensor):
#     if tensor.dim() == 0:
#         length = 1
#     elif tensor.dim() > 1:
#         raise ValueError('length for high dim tensor is undefined')
#     else:
#         length = len(tensor)
#     return length


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
