"""transforms_augs.py

Transforms and augmentations shared across dataloaders
"""
from typing import List, Optional, Sequence, Union
import torch
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur

import numpy as np
from skimage.exposure import equalize_adapthist
from skimage import filters
from skimage.transform import resize
from scipy.signal import convolve2d
from scipy import ndimage


class LineIntAugmentation(): 

    def __init__(self, **kwargs) -> None:
        self.test = kwargs.pop('test_mode', False)
        self.debug = kwargs.pop('debug', False)
        # self.test = True turns on some test mode behavior 
        # such taht when we sample a parameter range, it becomes
        # deterministically set to the center o the parameter range (.5 * (up - low))
        
        # self.op_on_intens = kwargs['op_on_intens']
    
    def _print(self, *args, **kwargs) -> None: 
        if self.debug: 
            print(*args, **kwargs)
         
    def __call__(self, x: np.ndarray): 
        raise NotImplementedError('Need to subclass LineIntAugmentation')
    
    def _to_intensity(self, x: np.ndarray) -> np.ndarray: 
        return 
    
    def _to_li(self, x: np.ndarray, I0=None) -> np.ndarray: 

        self._print('to_li: x input shape', x.shape)
        if len(x.shape) == 2: 
            x = x[None, :, :]

        # x[x < 1e-6] = 1e-6

        if I0 is None: 
            I0 = np.percentile(x, 99.5, axis=(1,2), keepdims=True)
        else: 
            self._print('I0.shape', I0.shape)
            I0 = np.array(I0)
            self._print('I0 shape', I0.shape)
        
        if len(I0.shape) != len(x.shape): 
            self._print('reshaping I0')
            I0 = I0.reshape(I0.shape + tuple([1 for _ in range(len(x.shape) - len(I0.shape))]))
        if I0.shape != x.shape: 
            self._print('tiling I0')
            I0 = np.tile(I0, [1, x.shape[1], x.shape[2]])
        # clip 
        x = np.clip(x / I0, 1e-6, 1)
        self._print('_to_li: x shape', x.shape)

        # convert to li 
        return -np.log(x)
    
    def sample_param(self, param: Union[float, Sequence], shape = None) -> np.ndarray: 

        if shape is None: 
            shape = [1]

        if isinstance(param, Sequence): 
            uni_rand = .5 * np.ones(shape) if self.test else torch.rand(shape).numpy()

            param = param[0] + uni_rand * (param[1] - param[0])
        else: 
            param = param * np.ones(shape, dtype=np.int_)
        
        return param


class AddDC(LineIntAugmentation): 

    def __init__(self, 
                rel_lower = .01, 
                rel_upper = .05, 
                prob_apply = 1, 
                op_on_intens = True, 
                const_bias=True, 
                debug=False, 
                **kwargs) -> None:
        super().__init__(**kwargs)
        self.op_on_intens = op_on_intens
        self.rel_low = rel_lower
        # self.rel_width = rel_upper - rel_lower
        self.rel_upper = rel_upper
        self.p = prob_apply
        self.const_bias = const_bias
        self.debug = debug
    
    def __call__(self, img: np.ndarray) -> np.ndarray:

        if torch.rand(1).item() > self.p: 
            return img

        if len(img.shape) == 2: 
            img = img[None, :, :]
        
        n_c = img.shape[0]

        # sample relative amount 
        # rel_amt = self.rel_low + self.rel_width * torch.rand(n_c).numpy()[:, None, None] # shape (n_c, 1, 1)
        rel_amt = self.sample_param([self.rel_low, self.rel_upper], (n_c, 1, 1))

        # exponentiate image 
        if self.op_on_intens: 
            proj = img
        else: 
            proj = np.exp(-img)

        if self.const_bias: 
            # dc_bias should be of shape [n_channels, 1, 1] so that we can broadcast
            dc_bias = rel_amt * np.percentile(proj, 1, axis=(1,2), keepdims=True) # type: np.ndarray
        else: 
            # here, dc_bias has full shape = proj.shape 
            dc_bias = torch.from_numpy(proj) # type: torch.Tensor
            # weight = torch.ones(1, 1, 10, 10).cuda()
            # weight /= weight.sum()
            gb = GaussianBlur((9, 9), sigma=30)

            if len(dc_bias.shape) == 3: 
                dc_bias = dc_bias[:, None, :, :]
            
            dc_bias = F.interpolate(dc_bias, [proj.shape[-2] // 3, proj.shape[-1] // 3], align_corners=True, antialias=True, mode='bilinear')

            for _ in range(10): 
                dc_bias = gb(dc_bias) # type: torch.Tensor
            
            dc_bias = F.interpolate(dc_bias, proj.shape[-2:], align_corners=True, antialias=True, mode='bilinear')
            dc_bias = gb(dc_bias)
            
            dc_bias = dc_bias.clip(min=0).squeeze(1).cpu().numpy() * rel_amt

        proj += dc_bias
        
        if self.op_on_intens: 
            li = proj
        else: 
            li = self._to_li(proj)

        if self.debug: 
            return li, dc_bias

        return li

       
class DynamicRewindow(LineIntAugmentation): 

    def __init__(self, 
                window_sz=50, 
                lower_q=3, 
                upper_q=96, 
                rescale=True, 
                **kwargs) -> None:
        super().__init__(**kwargs)
        self.window_sz = window_sz
        self.lower_q = lower_q 
        self.upper_q = upper_q
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """__call__

        Applies a dynamic rewindowing based on a window (with optionally randomly sampled size)

        The rewindowing is performed per channel; that is, if we randomly sample the size of the 
        window, then each channel of the resulting image will be subject to a different window. 

        The variability can be controlled by adjusting the width of [ window_sz[0], window_sz[1]]
        Args:
            img (np.ndarray): img to be rewindowed

        Returns:
            np.ndarray: rewindowed image
        """        

        # img = img.squeeze()
        # assume img is of shape [nchannels, h, w ]
        if len(img.shape) == 2: 
            img = img[None, :, :]
        
        self._print(img.shape)
        
        n_c = img.shape[0]

        # if isinstance(self.window_sz, Sequence) and len(self.window_sz) > 1: 
        #     window_sz = self.window_sz[0] + torch.rand(n_c).numpy() * (self.window_sz[1] - self.window_sz[0])
        # else: 
        #     window_sz = self.window_sz * np.ones(n_c, dtype=np.int_)
        window_sz = np.floor(self.sample_param(self.window_sz, n_c)).astype(np.int_)

        cen_win1, cen_win2 = img.shape[1] // 2, img.shape[2] // 2
        x_low, x_up = cen_win1 - window_sz, cen_win1 + window_sz
        y_low, y_up = cen_win2 - window_sz, cen_win2 + window_sz
        win_lower = np.zeros(n_c)
        win_upper = np.zeros(n_c)
        for ci, (xl, xu, yl, yu) in enumerate(zip(x_low, x_up, y_low, y_up)): 
            win_lower[ci] = np.percentile(img[ci, xl : xu, yl : yu], self.lower_q)
            win_upper[ci] = np.percentile(img[ci, xl : xu, yl : yu], self.upper_q)

        clipped = np.clip(img, win_lower[:, None, None], win_upper[:, None, None]) # type: np.ndarray
        clipped -= clipped.min(axis=(1,2), keepdims=True)
        clipped /= clipped.max(axis=(1,2), keepdims=True)

        for ci in range(n_c): 
            assert(clipped[ci].min() == 0)
            assert(clipped[ci].max() == 1)

        return clipped


class LineIntegral2Intensity(): 

    def __init__(self, 
                I0=1e4, 
                **kwargs) -> None:
        self.I0 = I0

    def __call__(self, img: np.ndarray): 
        return self.I0 * np.exp(-img)


class SaturateDetector(LineIntAugmentation): 
    def __init__(self, 
                 percentile_lower=93,
                 percentile_upper=99.9, 
                 op_on_intens=True, 
                **kwargs) -> None:
        super().__init__(**kwargs)
        self.ql = percentile_lower
        self.qu = percentile_upper
        self.op_on_intens = op_on_intens
    
    def __call__(self, img: np.ndarray):
        self._print('sat detector: input shape', img.shape)
        if len(img.shape) == 2: 
            img = img[None, ...]
        
        if self.op_on_intens: 
            proj = img 
        else: 
            proj = self._to_intensity(img)
        
        q = self.sample_param([self.ql, self.qu])
        I0_threshold = np.percentile(proj, q.squeeze(), axis=(1,2), keepdims=True)
        self._print('sat detector: q, I0_threshold shapes', q.shape, I0_threshold.shape, proj.shape)

        return self._to_li(proj, I0_threshold)


class SPR(LineIntAugmentation): 
    """SPR

    Performs spr / (1 + spr) correction on input image
    """
    def __init__(self, spr_low=0, spr_high=1.8, op_on_intens=True, prob_apply=1, **kwargs) -> None:

        super().__init__(**kwargs)

        self.spr_low = spr_low
        self.spr_high = spr_high
        self.op_on_intens = op_on_intens
        self.p = prob_apply

    def __call__(self, img: np.ndarray) -> np.ndarray:
        
        if torch.rand(1).item() > self.p: 
            return img

        # sample spr 
        # spr = self.spr_low + torch.rand(1).item() * (self.spr_range)
        spr = self.sample_param([self.spr_low, self.spr_high], 1)
        # spr = self.

        # assume img in shape (nb, nv, nu)
        prctls = np.percentile(img, 1, keepdims=True)

        #  (spr/(1+spr)) correction
        est_scat = (spr / (1 + spr)) * prctls
        img = img - est_scat

        # clip negative values, close to 0 values
        img[img < 1e-6] = 1e-6

        # convert back to line integral if that is the mode we're in 
        if not self.op_on_intens: 
            img = self._to_li(img)

        return img


class InjectQNoise(LineIntAugmentation):
    def __init__(self, 
                 F = .01, 
                 alpha = .8, 
                 I0=1e4, 
                 psf: Optional[Union[float, np.ndarray]] = None, 
                 debug=False,
                 op_on_intens=True,
                 prob_apply=1, 
                **kwargs) -> None:
        
        super().__init__(**kwargs)
        self.debug = debug
        self.p = prob_apply
        self.op_on_intens = op_on_intens

        # set F 
        self.F = F

        # set alpha 
        self.alpha = alpha

        # set conv size 
        if psf is not None:
            self.psf = psf
        
        else: 
            self.psf = np.array([
                                [0, 0, 0, 0, 0],
                                [0, 0.03, 0.06, 0.02, 0],
                                [0, 0.11, 0.98, 0.11, 0],
                                [0, 0.02, 0.06, 0.03, 0],
                                [0, 0, 0, 0, 0]
                            ], dtype=np.float64)
            self.psf = torch.from_numpy(self.psf)[None, None, ...]
                            
        # set I0
        self.I0 = I0

    def __call__(self, projection: np.ndarray) -> np.ndarray: 
        """__call__

        Args:
            projection (np.ndarray): input image -- should not have a batch dimension
        
        Injected quantum noise into projection data by exponentiating line-integral
        -> intensities, and adding poisson-distributed quantum noise w/ mean = intensity value
        at each pixel. 

        Returns:
            np.ndarray: noise-injected image
        """
        if torch.rand(1).item() > self.p: 
            return projection

        self._print('enter qnoise: ', projection.min(), projection.max())
        
        if len(projection.shape) == 2: 
            projection = projection[None, :, :]
        
        # n_b = projection.shape[0]

        if not self.op_on_intens: 
            # exponentiate projection
            projection = self._to_intensity(projection) * self.I0

        # obtain sigma
        # sigma_q_inj = np.sqrt(self.F * self.alpha * projection - self.alpha * self.alpha * self.F * projection)
        qF = self.sample_param([0, self.F]) # scalar qF
        sigma_q_inj = np.sqrt(qF * self.alpha * projection)
        # sigma_q_inj = 

        # sample white noise in N(0, 1)
        white_noise_q = np.random.randn(*projection.shape)

        # convolve
        if isinstance(self.psf, float): 
            injected_noise = ndimage.gaussian_filter(sigma_q_inj * white_noise_q, self.psf)
        elif isinstance(self.psf, (np.ndarray, torch.Tensor)): 
            pre_conv = torch.from_numpy(sigma_q_inj * white_noise_q)[:, None, :, :]
            injected_noise = F.conv2d(pre_conv, self.psf, padding='same').squeeze(1).numpy()
            # injected_noise = convolve2d(sigma_q_inj*white_noise_q, self.psf, mode='same')
        # injected_noise = ndimage.gaussian_filter(sigma_q_inj * white_noise_q, .58)

        # add back to image
        lds = np.maximum(1e-6, self.alpha * projection + injected_noise).astype(projection.dtype)
        self._print('qnoise inject', lds.shape)
        if self.op_on_intens: 
            return lds 

        return self._to_li(lds, self.I0)


class FlipChannels(): 

    def __init__(self, 
                prob_apply=0.5, 
                **kwargs) -> None:
        self.p = prob_apply
    def __call__(self, *arrs: List[np.ndarray]) -> List[np.ndarray]: 
        if torch.rand(1) > self.p: 
            return arrs 
        
        return_arrs = []
        for arr in arrs: 
            if len(arr.shape) == 2: 
                arr = arr[None, :, :]
            return_arrs.append(np.copy(arr[::-1, ...]))
        return return_arrs
        
class FlipVertical:
    def __init__(self, 
                prob_apply=0.5, 
                **kwargs):
        self.p = prob_apply

    def __call__(self, *arrs):
        """__call__

        Args:
            img (ndarray): assume img shape is (C,H,W)

        Returns:
            ndarray: the img, flipped along axis 1
        """

        if torch.rand(1) < self.p:
            flipped = [np.copy(np.flip(arr, axis=1)) for arr in arrs]
            return flipped 
            # return np.copy(np.flip(img, axis=1)), np.copy(np.flip(labels, axis=1))
        else:
            return arrs


class FlipHorizontal:
    def __init__(self, probability=0.5, 
                **kwargs):
        self.p = probability

    def __call__(self, *arrs):
        """__call__

        Args:
            img (ndarray): assume img shape is (C,H,W)

        Returns:
            ndarray: the img, flipped along axis 1
        """

        if torch.rand(1) < self.p:
            flipped = [np.copy(np.flip(arr, axis=2)) for arr in arrs]
            return flipped 
            # return np.copy(np.flip(img, axis=1)), np.copy(np.flip(labels, axis=1))
        else:
            return arrs


class RewindowFull(object): 
    def __init__(self, **kwargs) -> None:
        pass

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2: 
            img = img[None, ...]
        img -= img.min(axis=(1,2), keepdims=True)
        img /= img.max(axis=(1,2), keepdims=True)

        return img


class SideCropper():

    def __init__(self, axis=0, extent_max=20, test=False, **kwargs) -> None:
        self.axis = axis
        self.extent_max = extent_max
        self.test = test
    
    def __call__(self, *arrs: List[np.ndarray]) -> np.ndarray:
        # sample detector edge crop amount

        if self.test: 
            crop_left = self.extent_max // 2
            crop_right = self.extent_max - crop_left
        else: 
            crop_left = np.ceil(self.extent_max - 1 * torch.rand(1).item() ).astype(np.int_)
            crop_left = np.clip(crop_left, 1, self.extent_max - 1)
            crop_right = self.extent_max - crop_left

        if self.axis == 0: 
            return [np.copy(arr[:, crop_left : -crop_right, :]) for arr in arrs]
        elif self.axis == 1: 
            return [np.copy(arr[:, :, crop_left : -crop_right]) for arr in arrs]
        else: 
            raise RuntimeError('Incorrectly specified axis in SideCropper')
        

AUG_STR_MAP = {
    "LineIntegral2Intensity": LineIntegral2Intensity, 
    "AddDC": AddDC,    
    "DynamicRewindow": DynamicRewindow, 
    "FlipChannels": FlipChannels, 
    "FlipVertical": FlipVertical,
    "FlipHorizontal": FlipHorizontal,
    "InjectQNoise": InjectQNoise, 
    "SaturateDetector": SaturateDetector,
    "SideCropper": SideCropper, 
    "SPR": SPR,
    "RewindowFull": RewindowFull,
}