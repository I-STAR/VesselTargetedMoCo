"""seq_engine.py

Contains code wrapping training, validation loops + instantiation of model / criterion

"""
import os
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from networks.dataloaders.eval_metrics import mask_iou_torch 
from networks.unet.utils.criterion import SeqCriterion 
from networks.unet.utils.lr_schedulers import get_lr_lambda 
from .models.UNet import SeqUNetFrameConv 
from .utils.training_utils import (plot_training_val_curves, 
                                   make_training_figure, 
                                   separate_seq_gt, separate_seq_predictions) 


def get_model_criterion(device, 
                        model_config_kwargs: dict, 
                        optim_config_kwargs: Optional[dict] = None, 
                        crit_config_kwargs: Optional[dict] = None) -> Tuple[SeqUNetFrameConv, Optimizer, SeqCriterion, LambdaLR]: 
    
    if model_config_kwargs['model_class'] == 'SeqUNetFrameConv': 
        model = SeqUNetFrameConv(**model_config_kwargs)
    else: 
        raise ValueError(f"incorrectly specified model: got {model_config_kwargs['model_class']}")  
        
    if crit_config_kwargs: 
        criterion = SeqCriterion(crit_config_kwargs['bce_weight'], 
                                 crit_config_kwargs['bce_pos_weight'], 
                                 crit_config_kwargs['cl_dice_weight'], 
                                 crit_config_kwargs['cl_dice_iter'])
    
    if model_config_kwargs['distributed']: 
        model = nn.DataParallel(model).to(device)
        if crit_config_kwargs: 
            criterion = nn.DataParallel(criterion).to(device)
    else: 
        model = model.to(device)
        if crit_config_kwargs: 
            criterion = criterion.to(device)
    
    if not optim_config_kwargs: 
        return model

    if 'adamw' in optim_config_kwargs['optim'].lower(): 
        optimizer = optim.AdamW(model.parameters(), 
                                lr=optim_config_kwargs['lr'], 
                                weight_decay=optim_config_kwargs['weight_decay'])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("NUM TRAINABLE PARAMETERS:", num_params)

    lr_lambda = get_lr_lambda(optim_config_kwargs)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, verbose=False)
    
    return model, optimizer, criterion, scheduler


def train(
    model_config_kwargs: dict,
    optim_config_kwargs: dict,
    crit_config_kwargs: dict,
    epochs: int,
    dl_train: DataLoader,
    dl_valid: DataLoader,
    checkpoint_interval: int,
    model_save_loc: str,
    log_dir: str,
    accumulate_steps: int,
    aug_dict: dict,
) -> None:

    # initialize logging activities
    pred_preview_dir = os.path.join(log_dir, 'previews')
    os.makedirs(pred_preview_dir, exist_ok=True)

    # get the model
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    model, opt, criterion, sched = get_model_criterion(device, model_config_kwargs, optim_config_kwargs, crit_config_kwargs)
    print(torch.cuda.get_device_name(), device)

    start = time.perf_counter()

    # ==================== INITIALIZE METRIC STRUCTURES ====================
    global_step = 0
    training_losses = []
    val_losses = []
    val_ious_vasc = []
    val_ious_cath = []

    for epoch in range(epochs):

        print("epoch: %d / %d" % (epoch, epochs))

        model.train()

        for batch_idx, (x_batch, y_batch) in enumerate(dl_train):

            # print(x_batch.shape, y_batch.shape)
            if batch_idx % 100 == 0: 
                print(f'-- training batch {batch_idx} / {len(dl_train)}; lr: {sched.get_last_lr()}')

            x_batch = x_batch.to(device) # type: torch.Tensor
            y_batch = y_batch.to(device) # type: torch.Tensor
            
            raw_pred = model(x_batch) # type: torch.Tensor
            if torch.any(raw_pred.isnan()): 
                print('!!!! raw pred contains nan')

            loss = criterion(raw_pred, y_batch.float()) # type: torch.Tensor
            loss = loss.mean()
            loss.backward()

            total_norm = nn.utils.clip_grad_norm_(model.parameters(), 1e6)

            # accumulate gradients, step opt 
            if (global_step + 1) % accumulate_steps == 0:
                opt.step()
                opt.zero_grad()
                training_losses.append(loss.item())
            
            # step scheduler
            sched.step()

            if batch_idx % 100 == 0:
                outpath = os.path.join(pred_preview_dir, 'epoch-%d_train_batch-%d.png' % (epoch, batch_idx))

                # separate predictions for visualization 
                pred_vasc, pred_cath = separate_seq_predictions(torch.sigmoid(raw_pred).cpu().detach())

                # separate gt for visualization
                gt_vasc, gt_cath = separate_seq_gt(y_batch.cpu().detach(), mode='hinge')

                f = make_training_figure(x_batch.cpu().detach().numpy()[0],
                                         pred_vasc.numpy()[0], pred_cath.numpy()[0],
                                         gt_vasc = gt_vasc.numpy()[0], gt_cath=gt_cath.numpy()[0],
                                         outpath=outpath, 
                                         num_views=x_batch.shape[1]
                                         )

            global_step += 1

        print("finished training on %d batches" % len(dl_train), 'training_loss:', np.mean(training_losses[-len(dl_train):]))
        print("validation time!")

        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            iou_vasc = 0.0
            iou_cath = 0.0

            for idx_valid, (x_batch, y_batch) in enumerate(dl_valid):
                x_batch = x_batch.to(device)
                y_batch = y_batch.float().to(device)
                
                valid_pred = model(x_batch) # type: torch.Tensor
                if torch.any(valid_pred.isnan()): 
                    print('!!!! valid pred contains nan')
                
                valid_loss += criterion(valid_pred, y_batch.float()).mean().item() # type: torch.Tensor
                
                pred_v, pred_c = separate_seq_predictions(valid_pred.sigmoid())
                gt_v, gt_c = separate_seq_gt(y_batch.float())
 
                iou_vasc += torch.mean(mask_iou_torch(pred_v.flatten(0, 1), gt_v.flatten(0, 1))).item()
                iou_cath += torch.mean(mask_iou_torch(pred_c.flatten(0, 1), gt_c.flatten(0, 1))).item()

                pred_v = pred_v.cpu().detach()
                pred_c = pred_c.cpu().detach()
                gt_v = gt_v.cpu().detach()
                gt_c = gt_c.cpu().detach()

                if idx_valid % 75 == 0:
                    outpath = os.path.join(pred_preview_dir, 'epoch-%d_valid_batch-%d.png' % (epoch, idx_valid))
                    f = make_training_figure(x_batch.cpu().detach().numpy()[0],
                                             pred_v.numpy()[0], pred_c.numpy()[0],
                                             gt_vasc = gt_v.numpy()[0], gt_cath = gt_c.numpy()[0],
                                             outpath=outpath, 
                                             num_views=x_batch.shape[1]
                                            )

            print("validation loss", valid_loss / len(dl_valid))
            val_losses.append(valid_loss/len(dl_valid))
            
            print("validation iou vasc", iou_vasc / len(dl_valid))
            val_ious_vasc.append(iou_vasc/len(dl_valid))

            print("validation iou cath", iou_cath / len(dl_valid))
            val_ious_cath.append(iou_cath/len(dl_valid))

        # save the stuff so that we can visualize outside of tensorboard
        np.savez(os.path.join(model_save_loc, 'scalars.npz'), 
            training_losses=training_losses, 
            val_losses=val_losses, 
            val_ious_vasc=val_ious_vasc, 
            val_ious_cath=val_ious_cath)

        if (epoch % checkpoint_interval == 0) and (checkpoint_interval > 0):
            print("checkpointing")
            torch.save({"epoch": epoch, 
                        "model_state_dict": model.state_dict(), 
                        "optimizer_state_dict": opt.state_dict(), 
                        "epoch": epoch,
                        "aug_dict": aug_dict},
                os.path.join(model_save_loc, "checkpoint_%d.pth" % (epoch)))

            plot_training_val_curves(model_save_loc, training_losses, val_losses,
                                     val_ious_vasc = val_ious_vasc, 
                                     val_ious_cath = val_ious_cath, 
                                     loss_name = 'CL Dice + BCE Loss')

        print("time elapsed: %d m, %.3f s" % ((time.perf_counter() - start) // 60, (time.perf_counter() - start) % 60))
        print("----")

    return