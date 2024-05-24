"""get_aug_stack_config.py

Compartmentalizes a globally accessible hard-coding access to the augmentation
stack that we'll use for training. Enables reconfiguration for separate
train/val/test configs
"""

from typing import Any, Dict, Tuple
import copy


def get_aug_stacks() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]: 

    aug_stack = {
                "AddDC":   {'rel_lower': .01, 
                             'rel_upper': .05, 
                             'op_on_intens': True, 
                             'const_bias': True},
                "LineIntegral2Intensity":    {'I0': 1e4},
                "InjectQNoise":   {'F': .1, 
                             'alpha': .8, 
                             'I0': 3e4, 
                             'psf': None, 
                             'op_on_intens': True}, 
                "SaturateDetector":      {'percentile_lower': 90, 
                             'percentile_upper': 99.9, 
                             'op_on_intens': True},
                "SPR": {'prob_apply': 0.5,},
                "RewindowFull": {},
                # "DynamicRewindow": {'window_sz': [50, 100], 
                #              'lower_q': 3, 
                #              'upper_q': 96}, 
                'SideCropper': {'extent_max': 30, 
                             'test': True, 
                             'axis': 1}, 
                'FlipChannels': {'prob_apply': .15},
                'FlipVertical': {'prob_apply': 0.5},
            }
    
    test_val_update = {'FlipChannels': {'prob_apply': 0},
                       'FlipVertical': {'prob_apply': 0}, 
                       "SPR": {'prob_apply': 1,},
                        }

    valid_stack = copy.deepcopy(aug_stack)
    valid_stack.update(test_val_update)
    for k in valid_stack.keys(): 
        valid_stack[k]['test_mode'] = True
    
    test_stack = valid_stack.copy()

    return aug_stack, valid_stack, test_stack