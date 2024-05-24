"""criterion.py

"""
from typing import Optional
import torch 
import torch.nn as nn
import numpy as np

from .conn_loss import AlphaCLDice


class SeqCriterion(nn.Module): 

    def __init__(self, 
                 bce_weight, bce_pos_weight, 
                 cl_dice_weight, cl_dice_iter) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        if isinstance(bce_pos_weight, np.ndarray): 
            bce_pos_weight = torch.from_numpy(bce_pos_weight)
        
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight = bce_pos_weight)
        self.cl_dice_weight = cl_dice_weight
        self.cl_dice_loss = AlphaCLDice(iter=cl_dice_iter)
        
    def forward(self, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        # compute the centerline dice loss 
        # print(pred.shape, 'in seq criterion')
        if len(pred.shape) == 5: 
            loss_cld = self.cl_dice_loss(pred.sigmoid().flatten(0, 1), truth.flatten(0,1)) # type: torch.Tensor
        else: 
            loss_cld = self.cl_dice_loss(pred.sigmoid(), truth) # type: torch.Tensor
        
        loss_bce = self.bce_loss(pred, truth) # type: torch.Tensor
        
        if loss_cld.numel() == 1 and len(loss_cld.shape) ==0: 
            loss_cld = loss_cld.unsqueeze(0)
            loss_bce = loss_bce.unsqueeze(0)

        return self.cl_dice_weight * loss_cld + self.bce_weight * loss_bce 
