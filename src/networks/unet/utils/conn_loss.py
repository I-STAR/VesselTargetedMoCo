"""conn_loss.py

See: https://github.com/jocpae/clDice for reference

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_erode(img):
    if len(img.shape)==4:
        p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
        p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
        return torch.min(p1,p2)
    elif len(img.shape)==5:
        p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
        p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
        p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
        return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img):
    if len(img.shape)==4:
        return F.max_pool2d(img, (3,3), (1,1), (1,1))
    elif len(img.shape)==5:
        return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_):
    img1  =  soft_open(img)
    skel  =  F.relu(img-img1)
    for j in range(iter_):
        img  =  soft_erode(img)
        img1  =  soft_open(img)
        delta  =  F.relu(img-img1)
        skel  =  skel +  F.relu(delta-skel*delta)
    return skel


def soft_dice(pred: torch.Tensor, truth: torch.Tensor, threshold: float=.5, smooth: float=1.) -> torch.Tensor:
    """soft_dice

    Performs mean reduction by default 

    Expects pred, truth to be of shape (B,C,H,W)
    Args:
        y_true (torch.Tensor): [ground truth image]
        y_pred (torch.Tensor): [predicted image]
    Returns:
        torch.Tensor: [loss value]
    """

    # infer that it is (B,H,W) with no channel dimension; we would need to add channel axis
    if len(truth.size()) == 3: 
        truth = truth.unsqueeze(1)
    
    pred = pred.flatten(1) # (B,C,H,W) -> (B, C*H*W)
    truth = truth.flatten(1) # (B,C,H,W) -> (B, C*H*W)

    thresholded = (pred > threshold).float()

    truth = truth.float()

    intersection = torch.mul(thresholded, truth)
    numerator = 2 * intersection.sum(1) # sum starting at the channel dimension
    denominator = pred.sum(-1) + truth.sum(-1) 
    loss = 1 - (numerator + smooth) / (denominator + smooth)

    return loss.sum() / truth.shape[0]


class AlphaCLDice(nn.Module):
    def __init__(self, iter=3, alpha=0.5, smooth = 1.):
        super().__init__()
        self.iter = iter
        self.smooth = smooth
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        dice = soft_dice(pred, truth, smooth=self.smooth)

        skel_pred = soft_skel(pred, self.iter)
        skel_true = soft_skel(truth, self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, truth))+self.smooth)/(torch.sum(skel_pred)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)

        return (1.0 - self.alpha) * dice + self.alpha * cl_dice