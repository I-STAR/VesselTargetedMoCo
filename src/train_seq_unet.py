"""train_seq_unet.py

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
from networks.unet.seq_engine import train


def main(args: dict) -> None: 

    print(f'CURRENT MULTI FRAME SETTING: {args["multiframe"]}')

    model_save_dir = 'runs'
    trial_idx = date.today().strftime('%Y-%m-%d') + "-train"
    batch_sz = args['batch_size']
    accumulate_grad = 1
    epochs = args['epochs']
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

    [data_loader_train, 
     data_loader_val, 
     data_loader_test] = get_data_loaders(split_dict, 
                                      batch_size = batch_sz, 
                                      sanity=False, 
                                      augmentations=[aug_stack_train, aug_stack_valid, aug_stack_test], 
                                      multiclass='stack', 
                                      multiframe=args['multiframe'], 
                                      framestep=args['framestep'])

    model_config_kwargs = {
        'model_class': 'SeqUNetFrameConv', 
        'n_input_channels': 1, 
        'n_classes': 2, 
        'distributed': args['distributed'], 
        'bottleneck_dim': 200,
    }
    optim_config_kwargs = {
        'optim': 'adamw',
        'lr': 2e-6, 
        'weight_decay': 1e-2, # current default in adamw 
        'sched': 'const',
        'sched_kwargs': {
            'warmup_steps': 1000,
            'rate': .94, 
            'step_sz': len(data_loader_train) // 2,
            'min_lr': 1e-8,
        }
    }
    crit_config_kwargs = {
        'bce_weight': .5, 
        'bce_pos_weight': np.ones((1, 440, 546)) * 2.75, 
        'cl_dice_weight': .3, 
        'cl_dice_iter': 15, 
    }

    print("--------- MODEL CONFIG KWARGS ----------")
    pprint(model_config_kwargs)
    print("--------- OPTIM CONFIG KWARGS ----------")
    pprint(optim_config_kwargs)
    pprint("--------- CRIT CONFIG KWARGS ----------")
    pprint(crit_config_kwargs)
    
    train(model_config_kwargs, optim_config_kwargs, crit_config_kwargs,
         epochs, data_loader_train, data_loader_val, 
         checkpoint_interval, 
         model_save_loc, log_dir, 
         accumulate_grad, 
         aug_stack_train,
    )
    
    return 


def parse_commandline():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--multiframe', type=int, default=1)
    parser.add_argument('--framestep', type=int, default=1)
    parser.add_argument('--distributed', action='store_true')
    return vars(parser.parse_args())


if __name__ == "__main__":
    main(parse_commandline())