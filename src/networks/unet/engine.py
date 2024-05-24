"""engine.py


"""
import os
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from networks.dataloaders.eval_metrics import mask_iou_torch, mask_pr_torch
from networks.unet.utils.criterion import SeqCriterion
from .models.UNet import UNetNew
from .utils.training_utils import (plot_training_val_curves, 
                                  separate_gt, separate_predictions, 
                                  make_training_figure)


def get_model(device, model_config_kwargs: dict, optimizer="SGD", instance_segmentation = False, params=None):

    model = UNetNew(1, 2, bilinear=True, **model_config_kwargs).to(device)
    if optimizer == "SGD":
        opt = optim.SGD(
            model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.001
        )
    elif optimizer == "adam":
        opt = optim.Adam(model.parameters(), lr=5e-4)
    
    elif optimizer == "adamw": 
        print('using AdamW')
        opt = optim.AdamW(model.parameters(), lr=1e-4)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("NUM TRAINABLE PARAMETERS:", num_params)

    return model, opt


def train(
    model_config_kwargs: dict,
    epochs: int,
    dl_train: DataLoader,
    dl_valid: DataLoader,
    checkpoint_interval: int,
    model_save_loc: str,
    log_dir: str,
    accumulate_steps: int,
) -> None:

    # initialize logging activities
    pred_preview_dir = os.path.join(log_dir, 'previews')
    os.makedirs(pred_preview_dir, exist_ok=True)

    # get the model
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    model, opt = get_model(device, model_config_kwargs, optimizer="adamw")
    print(torch.cuda.get_device_name())
    print(device)

    start = time.perf_counter()


    # ==================== INITIALIZE THE LOSS FUNCTION ====================
    # loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=7.25 * torch.ones((1, 440, 546)).to(device))
    # loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=2.75 * torch.ones((1, 440, 546)).to(device))
    loss_fn = SeqCriterion(0.5, np.ones((1, 440, 546)) * 2.75, .3, 9, None, None).to(device)
    
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
            
            if batch_idx % 100 == 0: 
                print(f'-- training batch {batch_idx} / {len(dl_train)}')

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            raw_pred = model(x_batch)

            loss = loss_fn(raw_pred, y_batch.float()) # type: torch.Tensor
            loss.backward()

            # accumulate gradients
            if (global_step + 1) % accumulate_steps == 0:
                opt.step()
                opt.zero_grad()
                training_losses.append(loss.item())

            if batch_idx % 100 == 0:
                outpath = os.path.join(pred_preview_dir, 'epoch-%d_train_batch-%d.png' % (epoch, batch_idx))

                # separate predictions for visualization 
                pred_vasc, pred_cath = separate_predictions(torch.sigmoid(raw_pred).cpu().detach())

                # separate gt for visualization
                gt_vasc, gt_cath = separate_gt(y_batch.cpu().detach(), mode='hinge')

                f = make_training_figure(x_batch.cpu().detach().numpy()[0],
                                         pred_vasc.numpy()[0], pred_cath.numpy()[0],
                                         gt_vasc = gt_vasc.numpy()[0], gt_cath=gt_cath.numpy()[0],
                                         outpath=outpath
                                         )

            global_step += 1

        print("finished training on %d batches" % len(dl_train))
        print("validation time!")

        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            iou_vasc = 0.0
            iou_cath = 0.0

            for idx_valid, (x_batch, y_batch) in enumerate(dl_valid):
                x_batch = x_batch.to(device)
                y_batch = y_batch.float().to(device)

                valid_pred = model(x_batch)
                valid_loss += loss_fn(valid_pred, y_batch.float()).item()

                # separate predictions for visualization 
                pred_vasc, pred_cath = separate_predictions(torch.sigmoid(valid_pred).cpu().detach())

                # separate gt for visualization
                gt_vasc, gt_cath = separate_gt(y_batch.cpu().detach(), mode='hinge')
                
                iou_vasc += torch.mean(mask_iou_torch(pred_vasc, gt_vasc.float())).item()
                iou_cath += torch.mean(mask_iou_torch(pred_cath, gt_cath.float())).item()

                if idx_valid % 75 == 0:

                    outpath = os.path.join(pred_preview_dir, 'epoch-%d_valid_batch-%d.png' % (epoch, idx_valid))
                    f = make_training_figure(x_batch.cpu().detach().numpy()[0],
                                             pred_vasc.numpy()[0], pred_cath.numpy()[0],
                                             gt_vasc = gt_vasc.numpy()[0], gt_cath = gt_cath.numpy()[0],
                                             outpath=outpath
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

        if epoch % checkpoint_interval == 0:
            print("checkpointing")
            torch.save({"epoch": epoch, 
                        "model_state_dict": model.state_dict(), 
                        "optimizer_state_dict": opt.state_dict(), 
                        "epoch": epoch,},
                os.path.join(model_save_loc, "checkpoint_%d.pth" % (epoch)))

            plot_training_val_curves(model_save_loc, training_losses, val_losses, val_ious_vasc, val_ious_cath)

        print("time elapsed: %d m, %.3f s" % ((time.perf_counter() - start) // 60, (time.perf_counter() - start) % 60))
        print("----")

    return