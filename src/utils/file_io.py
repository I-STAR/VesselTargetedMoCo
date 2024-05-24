"""file_io.py

"""
from typing import Optional, Tuple, Sequence

import h5py
import numpy as np


def h5_save(save_path, forward_vol, forward_mask):

    with h5py.File(save_path, 'w') as hf:
        hf.create_dataset('forward_vol', data=forward_vol)
        hf.create_dataset('forward_mask', data=forward_mask)

    return


def h5_load(load_path) -> Tuple[np.ndarray, np.ndarray]:

    with h5py.File(load_path, 'r') as hf:
        return hf.get('forward_vol')[()], hf.get('forward_mask')[()]


def h5_multi_load(save_path):

    with h5py.File(save_path, 'r') as hf:
        contents = {}
        for k in hf.keys():
            contents[k] = hf.get(k)[()]

    return contents


def h5_multi_save(save_path, **kwargs):

    with h5py.File(save_path, 'w') as hf:
        for k, v in kwargs.items():
            hf.create_dataset(k, data=v)

    return
