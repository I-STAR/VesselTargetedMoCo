"""train_unet.py

"""
import argparse
import datetime
from datetime import date
import os
from pprint import pprint
from typing import Tuple 

import numpy as np
from networks.dataloaders import get_aug_stack_config
from networks.dataloaders.sequence_loader import get_data_loaders
from networks.unet.engine import train
from networks.unet.models.UNet import config_model


def config_from_cl() -> Tuple[dict, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bn_config', default=0, type=int)

    args = vars(parser.parse_args())
    
    model_config_kwargs, run_id_str = config_model(args['bn_config'])
    
    return model_config_kwargs, run_id_str 


def main():

    model_config_kwargs, append_str = config_from_cl()

    model_save_dir = 'ablation_runs'
    # trial_idx = date.today().strftime('%Y-%m-%d') + "-train_bce_1frame" + append_str
    trial_idx = date.today().strftime('%Y-%m-%d') + "-train_bce_cldice_1frame" + append_str

    batch_sz = 24
    accumulate_grad = 1
    epochs = 80
    checkpoint_interval = 1 

    split_dict_path = ''
    split_dict = np.load(split_dict_path)
    
    now = datetime.datetime.now()
    log_dir = os.path.join(
        model_save_dir,
        "%s_training" % (trial_idx),
        now.strftime("%Y%m%d-%H%M%S"),
    )

    model_save_loc = os.path.join(model_save_dir, trial_idx)
    os.makedirs(model_save_loc, exist_ok=True)

    aug_stack_train, aug_stack_valid, aug_stack_test = get_aug_stack_config.get_aug_stacks()

    for augs in [aug_stack_train, aug_stack_valid, aug_stack_test]: 
        augs.pop('FlipChannels')
    
    print('TRAINING AUGMENTATION STACK SETTINGS')
    pprint(aug_stack_train)
    [data_loader_train, 
     data_loader_val, 
     data_loader_test] = get_data_loaders(split_dict, 
                                      batch_size = batch_sz, 
                                      sanity=False, 
                                      augmentations=[aug_stack_train, aug_stack_valid, aug_stack_test], 
                                      multiclass='stack')    
    print('STARTING TRAINING')
    train(
        model_config_kwargs,
        epochs,
        data_loader_train,
        data_loader_val,
        checkpoint_interval,
        model_save_loc,
        log_dir,
        accumulate_grad,
    )

    return


if __name__ == "__main__":
    main()