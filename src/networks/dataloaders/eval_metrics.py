"""eval_metrics.py

"""
from typing import List, Sequence, Tuple, Union, Optional
from time import perf_counter


import torch
import torch.functional as F
import torch.nn as nn
import numpy as np
from scipy.ndimage.morphology import binary_dilation
from skimage.measure import label


def mask_dice_np(pred, truth, threshold=.5) -> np.ndarray: 
    """mask_dice_np

    Args:
        pred (ndarray): prediction tensor (B,C,H,W)
        truth (ndarray): ground truth tensor (B,C,H,W)
        threshold (float, optional): threshold for prediction. Defaults to .5.
    """

    # print('size of pred', pred.size())
    # print('size of truth', truth.size())

    if len(pred.shape) != 4:
        # print('did not find batch dimension. Adjusting sum axes')
        sum_axes = (1, 2)
    else:
        sum_axes = (2, 3)

    thresholded = np.where(pred > threshold, 1, 0).astype(np.float64)
    truth = truth.astype(np.float64)

    # should be size (B,C,H,W)
    intersection = thresholded * truth

    # print('size of intersection:', intersection.size())

    # sum over H, W: (B,C,H,W) -> (B,C)
    pred_pos = np.sum(thresholded, axis=sum_axes)
    truth_pos = np.sum(truth, axis=sum_axes)

    # print('pred pos size', pred_pos.size())
    # print('truth_pos size', truth_pos.size())

    intersection = np.sum(intersection, axis=sum_axes)

    dice = 2 * intersection / (pred_pos + truth_pos)

    return dice


def mask_iou_np(pred, truth, threshold=.5, collapse=False) -> np.ndarray:
    """mask_iou_np

    Args:
        pred (ndarray): prediction tensor (B,C,H,W)
        truth (ndarray): ground truth tensor (B,C,H,W)
        threshold (float, optional): threshold for prediction. Defaults to .5.
    """

    # print('size of pred', pred.size())
    # print('size of truth', truth.size())

    if len(pred.shape) != 4:
        # print('did not find batch dimension. Adjusting sum axes')
        sum_axes = (1, 2)
    else:
        sum_axes = (2, 3)

    thresholded = np.where(pred > threshold, 1, 0).astype(np.float64)
    truth = truth.astype(np.float64)

    # should be size (B,C,H,W)
    intersection = thresholded * truth

    # print('size of intersection:', intersection.size())

    # sum over H, W: (B,C,H,W) -> (B,C)
    pred_pos = np.sum(thresholded, axis=sum_axes)
    truth_pos = np.sum(truth, axis=sum_axes)

    # print('pred pos size', pred_pos.size())
    # print('truth_pos size', truth_pos.size())

    intersection = np.sum(intersection, axis=sum_axes)

    iou = intersection / (pred_pos + truth_pos - intersection)

    return iou


def mask_iou_torch(pred: torch.Tensor, truth: torch.Tensor, threshold: float =.5, collapse=False) -> torch.Tensor:
    """mask_iou_torch

    Args:
        pred (torch.Tensor): prediction tensor (B,C,H,W)
        truth (torch.Tensor): ground truth tensor (B,C,H,W)
        threshold (float, optional): threshold for prediction. Defaults to .5.
    """

    # print('size of pred', pred.size())
    # print('size of truth', truth.size())

    if len(truth.size()) == 3: 
        # infer that it is (B, H, W) 
        truth = truth.unsqueeze(1)

    thresholded = (pred > threshold).float()
    truth = truth.float()

    # should be size (B,C,H,W)
    intersection = torch.mul(thresholded, truth)

    # print('size of intersection:', intersection.size())

    # sum over H, W: (B,C,H,W) -> (B,C)
    pred_pos = thresholded.sum((2, 3))
    truth_pos = truth.sum((2, 3))

    # print('pred pos size', pred_pos.size())
    # print('truth_pos size', truth_pos.size())

    intersection = intersection.sum((2, 3))

    iou = intersection / (pred_pos + truth_pos - intersection)

    return iou


def mask_pr_torch(pred: torch.Tensor, truth: torch.Tensor, threshold: float=.5) -> Tuple[torch.Tensor, torch.Tensor]: 
    """mask_pr_torch

    Args:
        pred (torch.Tensor): prediction
        truth (torch.Tensor): trtuh tensor (B,C,H,W)
        threshold (float, optional): where to threshold predictions. Defaults to .5.

    Returns: 
        Tuple[torch.Tensor]: the precision and the recall tensors
    """

    if len(truth.size()) == 3: 
        # infer that it is (B, H, W) 
        truth = truth.unsqueeze(1)

    # thresholded = (pred > threshold).float()
    thresholded = pred
    truth = truth.float()

    # should be size (B,C,H,W)
    intersection = torch.mul(thresholded, truth)

    # print('size of intersection:', intersection.size())

    # sum over H, W: (B,C,H,W) -> (B,C)
    truth_pos = truth.sum((2, 3))
    pred_pos = thresholded.sum((2,3))

    intersection = intersection.sum((2, 3))

    precision = intersection / (pred_pos)
    recall = intersection / (truth_pos)

    return precision, recall


def mask_pr_np(pred, truth, threshold=.5) -> Tuple[np.ndarray, np.ndarray]:
    """mask_pr_np

    Args:
        pred (ndarray): prediction tensor (B,C,H,W)
        truth (ndarray): ground truth tensor (B,C,H,W)
        threshold (float, optional): threshold for prediction. Defaults to .5.

    Returns: 
        Tuple[np.ndarray]
    """
    if len(pred.shape) != 4:
        # print('did not find batch dimension. Adjusting sum axes')
        sum_axes = (1, 2)
    else:
        sum_axes = (2, 3)

    thresholded = np.where(pred > threshold, 1, 0).astype(np.float64)
    truth = truth.astype(np.float64)

    # should be size (B,C,H,W)
    intersection = thresholded * truth

    # sum over H, W: (B,C,H,W) -> (B,C)
    pred_pos = np.sum(thresholded, axis=sum_axes)
    truth_pos = np.sum(truth, axis=sum_axes)

    intersection = np.sum(intersection, axis=sum_axes)

    recall = intersection / truth_pos
    precision = intersection / pred_pos

    return precision, recall


def mask_dilated_prec(pred: np.ndarray, truth: np.ndarray, threshold=.5) -> np.ndarray: 
    """mask_dilated_prec

    Computes the precision with respect to a dilated version of the 
    ground truth. 

    Assumes that each of the arrays is already binarized. 

    Assumes that each of pred, truth are np arrays with shape 
    (n_views, h, w)

    Args:
        pred (np.ndarray): _description_
        truth (np.ndarray): _description_
        threshold (float, optional): _description_. Defaults to .5. NOTE: DEPRECATED

    Returns:
        np.ndarray: the precision
    """
    if len(truth.shape) == 4: 
        # check if there is a 1 channel dimension
        truth = truth.squeeze()
    if len(pred.shape) == 4: 
        pred = pred.squeeze()

    dil_gt = np.zeros_like(truth)

    # dilate the truth with a 3x3 connectivity kernel 
    # print('beginning dilation')
    start = perf_counter()
    for b_i in range(truth.shape[0]): 
        dil_gt[b_i] = binary_dilation(truth[b_i], structure=np.ones((3,3)))

    # print("time elapsed: %d m, %.3f s" % ((perf_counter() - start) // 60, 
    #                                         (perf_counter() - start) % 60))
    # print("----")

    # TODO: remove this print statement
    # print(dil_gt.dtype, dil_gt.sum(), truth.sum())

    # compute the precision as normal
    pp, _ = mask_pr_np(pred, dil_gt)

    return pp


def mask_recall_np(pred: np.ndarray, truth: np.ndarray, threshold=.5) -> np.ndarray:
    """mask_recall_np

    Args:
        pred (ndarray): prediction tensor (B,C,H,W)
        truth (ndarray): ground truth tensor (B,C,H,W)
        threshold (float, optional): threshold for prediction. Defaults to .5.

    Returns: 
        recall (np.ndarray): the computed recall (TP / (positive in truth))
    """
    if len(pred.shape) != 4:
        # print('did not find batch dimension. Adjusting sum axes')
        sum_axes = (1, 2)
    else:
        sum_axes = (2, 3)

    thresholded = np.where(pred > threshold, 1, 0).astype(np.float64)
    truth = truth.astype(np.float64)

    # should be size (B,C,H,W)
    intersection = thresholded * truth

    # sum over H, W: (B,C,H,W) -> (B,C)
    # pred_pos = np.sum(thresholded, axis=sum_axes)
    truth_pos = np.sum(truth, axis=sum_axes)

    intersection = np.sum(intersection, axis=sum_axes)

    # iou = intersection / (pred_pos + truth_pos - intersection)

    recall = intersection / truth_pos

    return recall


def mask_precision_np(pred: np.ndarray, truth: np.ndarray, threshold=.5) -> np.ndarray:
    """mask_recall_np

    Args:
        pred (ndarray): prediction tensor (B,C,H,W)
        truth (ndarray): ground truth tensor (B,C,H,W)
        threshold (float, optional): threshold for prediction. Defaults to .5.

    Returns: 
        recall (np.ndarray): the computed recall (TP / (positive in truth))
    """
    if len(pred.shape) != 4:
        # print('did not find batch dimension. Adjusting sum axes')
        sum_axes = (1, 2)
    else:
        sum_axes = (2, 3)

    thresholded = np.where(pred > threshold, 1, 0).astype(np.float64)
    truth = truth.astype(np.float64)
    intersection = thresholded * truth

    # sum over H, W: (B,C,H,W) -> (B,C)
    pred_pos = np.sum(thresholded, axis=sum_axes)
    intersection = np.sum(intersection, axis=sum_axes)

    precision = intersection / pred_pos

    return precision



def mask_recall_torch(pred: torch.Tensor, truth: torch.Tensor, threshold: float=.5) -> torch.Tensor: 
    """mask_recall_torch

    Args:
        pred (torch.Tensor): prediction
        truth (torch.Tensor): trtuh tensor (B,C,H,W)
        threshold (float, optional): where to threshold predictions. Defaults to .5.
    """

    if len(truth.size()) == 3: 
        # infer that it is (B, H, W) 
        truth = truth.unsqueeze(1)

    thresholded = (pred > threshold).float()
    truth = truth.float()

    # should be size (B,C,H,W)
    intersection = torch.mul(thresholded, truth)

    # print('size of intersection:', intersection.size())

    # sum over H, W: (B,C,H,W) -> (B,C)
    truth_pos = truth.sum((2, 3))

    intersection = intersection.sum((2, 3))

    recall = intersection / (truth_pos)

    return recall
