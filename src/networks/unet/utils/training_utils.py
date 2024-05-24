"""training_utils.py

"""
import os 
from typing import Tuple, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors as mplcolors


COLORS = ['yellow', 
          'tomato', 
          'lime', 
          'cyan', 
          'forestgreen', 
          'darkviolet', 
          'forestgreen', 
          'fuchsia', 
          'dodgerblue', 
        ]


def colormap_binary(fg, bg=(0,0,0), alpha=None):
    fg = mplcolors.to_rgb(fg)
    bg = mplcolors.to_rgb(bg)
    cmap = mplcolors.LinearSegmentedColormap.from_list('Binary', (bg,fg), 256)
    if alpha is not None:
        cmap._init()
        cmap._lut[:,-1] = np.linspace(0, alpha, cmap.N+3)
    return cmap


def squeeze_out_channel_dim(fp, pred_vasc, pred_cath, gt_vasc, gt_cath) -> Tuple[np.ndarray]:
    fp = fp.squeeze(1)
    pred_vasc = pred_vasc.squeeze(1) 
    if pred_cath is not None: 
        pred_cath = pred_cath.squeeze(1)
    if gt_vasc is not None: 
        gt_vasc = gt_vasc.squeeze(1)
    if gt_cath is not None: 
        gt_cath = gt_cath.squeeze(1)
    return fp, pred_vasc, pred_cath, gt_vasc, gt_cath



def make_training_figure(fp: np.ndarray, 
                         pred_vasc: np.ndarray, 
                         pred_cath: Optional[np.ndarray] = None, 
                         gt_vasc: Optional[np.ndarray] = None, 
                         gt_cath: Optional[np.ndarray] = None, 
                         outpath: Optional[str] = None, 
                         downsample=2, num_views=5, preview=False) -> plt.figure:

    """make_training_figure

    Make a figure displaying the forward projections and the associated predictions / ground truths

    Optionally: 
    - can write out the figure for easy previewing 
    - 

    Args:
        fp (np.ndarray): np array of forward projections to be generated into a figure
        pred_vasc(np.ndarray): vascular predictions
        pred_cath(np.ndarray): catheter predictions
        outpath (Optional[str], optional): optional filepath for saving the figure. Defaults to None.
        labels (np.ndarray, optional): optional labels to include for vasculature. Defaults to None.
        gt_vasc (np.ndarray, optional): optional vasc labels to include. Defaults to None.
        gt_cath (np.ndarray, optional): optional cath labels to include. Defaults to None.
        downsample (int, optional): extent of downsampling image (crude). Defaults to 10.
        num_views (int, optional): number of views to include in the figure panel. Defaults to 5.
        preview (bool, optional): whether to show the pyplot window. Defaults to False.

    Returns: 
        f, the generated preview figure object
    """
    
    if len(fp.shape) == 4: 
        fp, pred_vasc, pred_cath, gt_vasc, gt_cath = squeeze_out_channel_dim(fp, pred_vasc, pred_cath, gt_vasc, gt_cath)

    if gt_vasc is not None: 
        if len(gt_vasc.shape) == 2: 
            gt_vasc = gt_vasc[None, :,:]
    if gt_cath is not None: 
        if len(gt_cath.shape) == 2: 
            gt_cath = gt_cath[None, :,:]

    # close all existing figures to avoid overflow warnings
    plt.close('all')

    # fig shape
    nproj, ydim, xdim = fp[:, ::downsample, ::downsample].shape

    # set the number of views to be at most the number of projections
    if num_views > nproj: 
        num_views = nproj

    # set the grid for populating the images
    grid = np.zeros((ydim, xdim * num_views))
    
    # populate the main grid that contains the input images (fp)
    view_spacing = nproj // num_views
    for grid_pos, proj_idx in enumerate(range(0, nproj, view_spacing)):
        grid[:, xdim*grid_pos:xdim * (grid_pos + 1)] = fp[proj_idx,
                                                          ::downsample,
                                                          ::downsample]

    # calculate the number of panels based on whether we have provided
    # the ground truth. Default is 2: one for input, one for pred.
    nfig_rows = 2 
    # add 1 if we have separate cath pred
    nfig_rows += (pred_cath is not None) 
    # add one again if we have ground truth definition
    nfig_rows += ((gt_vasc is not None) or (gt_cath is not None))

    # generate the figure. Makes use of the grid size to 
    # ensure that the aspect ratio / figsize are appropriate.

    aspect_ratio = grid.shape[1] / grid.shape[0]
    f, axes = plt.subplots(nfig_rows, 1,
                           figsize=(3.5*aspect_ratio, 3.5*nfig_rows),
                           facecolor='black',
                           frameon=False)
    
    # print("aspect_ratio:", aspect_ratio, 'nfig_rows', nfig_rows)
    axes[0].imshow(grid, origin='lower', cmap='gray', vmin=grid.min(), vmax=grid.max())
    axes[0].axis('off')

    # show the predictions
    pred_grid = np.zeros((2, ydim, xdim * num_views))
    for grid_pos, proj_idx in enumerate(range(0, nproj, view_spacing)):
        pred_grid[0, :, xdim*grid_pos:xdim * (grid_pos + 1)] = pred_vasc[proj_idx, ::downsample, ::downsample]
        
        if pred_cath is not None: 
            pred_grid[1, :, xdim*grid_pos:xdim * (grid_pos + 1)] = pred_cath[proj_idx, ::downsample, ::downsample]
                            
    axes[1].imshow(pred_grid[0], origin='lower', cmap='gray', vmin=0, vmax=1)
    axes[1].axis('off')

    if pred_cath is not None: 
        axes[2].imshow(pred_grid[1], origin='lower', cmap='gray', vmin=0, vmax=1)
        axes[2].axis('off')

    # plot the ground truth contours over the input image, if ground truth
    # labels are provided 
    underlying = False
    if gt_vasc is not None:
        axes[-1].imshow(grid, origin='lower', cmap='gray')
        underlying = True
        grid2 = np.zeros((ydim, xdim * num_views))
        for grid_pos, proj_idx in enumerate(range(0, nproj, view_spacing)):
            grid2[:,
                  xdim*grid_pos:xdim * (grid_pos + 1)] = gt_vasc[proj_idx, ::downsample, ::downsample]

        CS = axes[-1].contour(grid2,
                             origin='lower',
                             cmap=colormap_binary('yellow', alpha=1))

    if gt_cath is not None:
        if not underlying:
            axes[-1].imshow(grid, origin='lower', cmap='gray')
        grid3 = np.zeros((ydim, xdim * num_views))
        for grid_pos, proj_idx in enumerate(range(0, nproj, view_spacing)):
            grid3[:,
                  xdim*grid_pos:xdim * (grid_pos + 1)] = gt_cath[proj_idx, ::downsample, ::downsample]

        CS = axes[-1].contour(grid3,
                             origin='lower',
                             cmap=colormap_binary('limegreen', alpha=1))

    axes[-1].axis('off')

    plt.axis('off')
    f.subplots_adjust(wspace=0, hspace=0)

    if outpath is not None:
        f.savefig(outpath, facecolor='black',
                  bbox_inches='tight', transparent=True)

    if preview:
        plt.show()

    return f


def separate_predictions(raw_pred: torch.Tensor, ) -> Tuple[torch.Tensor, torch.Tensor]:

    b, c, h, w = list(raw_pred.size())  # should be b, 40, h, w

    # super super ratchet

    return raw_pred[:, :c//2, :, :], raw_pred[:, c//2:, :, :]


def separate_seq_predictions(raw_pred: torch.Tensor, ) -> Tuple[torch.Tensor, torch.Tensor]: 
    b, n_frames, c, h, w = list(raw_pred.size())  # should be b, 40, h, w

    # super super ratchet

    return raw_pred[:, :, :c//2, :, :], raw_pred[:, :, c//2:, :, :]


def separate_seq_gt(gt: torch.Tensor, mode='hinge') -> Tuple[torch.Tensor, torch.Tensor]: 
    if mode == 'hinge': 
        b, nf, c, h, w = list(gt.size())
        return gt[:,:, :c//2], gt[:,:, c//2:]
    
    elif mode == 'softmax': 
        return torch.where(gt == 1, 1, 0), torch.where(gt == 2, 1, 0)
    
    else: 
        raise ValueError('mode %s not supported in separate_gt' % mode)


def separate_gt(gt: torch.Tensor, mode='hinge') -> Tuple[torch.Tensor, torch.Tensor]: 

    if mode == 'hinge': 
        b, c, h, w = list(gt.size())
        return gt[:, :c//2], gt[:, c//2: ]
    
    elif mode == 'softmax': 
        return torch.where(gt == 1, 1, 0), torch.where(gt == 2, 1, 0)
    
    else: 
        raise ValueError('mode %s not supported in separate_gt' % mode)


def plot_training_val_curves(save_dir: str, 
                             train_losses: np.ndarray,
                             val_losses: np.ndarray, 
                             val_ious_vasc: Optional[np.ndarray] = None, 
                             val_ious_cath: Optional[np.ndarray] = None, 
                             loss_name: Optional[str] = 'BCE w/ Logits Loss') -> None:

    #TODO: need to make x-axes for training, validation losses same scale

    # path sanity
    os.makedirs(save_dir, exist_ok=True)

    # plot figure
    plt.style.use('dark_background')
    plt.figure(figsize=(5, 3))
    
    # plot validation losses
    # infer the number of iterations per epoch 
    iter_per_epoch = len(train_losses) / len(val_losses)
    plt.plot(np.arange(len(train_losses)) / iter_per_epoch, train_losses, color=COLORS[0], label='training losses', lw=3)
    plt.plot(np.arange(1, len(val_losses) + 1), val_losses, color=COLORS[1], label='valid losses', lw=3)
        
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel(loss_name)
    plt.ylim([0, np.max(val_losses) * 1.5])
    plt.title('Training and Validation Loss Curves')

    plt.savefig(os.path.join(save_dir, 'train_val_loss_curve.png'))

    # ==================== second training figure ========================
    if (val_ious_vasc is not None) and (val_ious_cath is not None): 
        plt.figure(figsize=(5, 3))
        plt.plot(val_ious_vasc, color=COLORS[0], label='val iou vasc', lw=3)
        plt.plot(val_ious_cath, color=COLORS[1], label='val iou cath', lw=3)
            
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('IOU')
        plt.ylim([0, 1])
        plt.title('Validation IOU Curves')
        plt.savefig(os.path.join(save_dir, 'Validation IOU Curves'))

    return 