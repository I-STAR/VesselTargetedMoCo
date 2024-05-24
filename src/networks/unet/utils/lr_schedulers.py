"""lr_schedulers.py

Defines various lamdba functions for lr scheduling


"""
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from functools import partial


def const_fn(step: int): 
    return 1


def eval_lr(step, warmup_steps=3000, rate=.99, step_sz=3000, min_lr=1e-6, **kwargs): 
    if step <= warmup_steps: 
            return max((step / warmup_steps), min_lr) # linear ramp up to full learning rate
    else: 
        return max(rate ** (step // step_sz), min_lr)


def get_lr_lambda(optim_config_kwargs: Dict[str, Any]) -> Callable[[int], float]: 
    
    if optim_config_kwargs['sched'].lower() == 'warmup_decay': 
        return partial(eval_lr, **optim_config_kwargs['sched_kwargs'])
    else: # if optim_config_kwargs['sched'].lower() == 'const': 
        return const_fn