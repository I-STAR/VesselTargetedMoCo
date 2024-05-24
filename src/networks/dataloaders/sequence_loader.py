"""sequence_loader.py

"""
import os 
from typing import Callable, Dict, List, Optional

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from utils.file_io import h5_load, h5_multi_load 
from .transforms_augs import *


def get_augmentations(aug_dict: dict) -> Dict:

    aug_instances = {}
    for augment, aug_kwargs in aug_dict.items():
        AugClass = AUG_STR_MAP[augment] 
        aug_instances[augment] = AugClass(**aug_kwargs)
    return aug_instances


def get_data_loaders(
    split_dict,
    batch_size=1,
    transform=None, target_transform=None,
    augmentations=None,
    prefix=None,
    shuffle_train=True, shuffle_valid=True, shuffle_test=False,
    num_workers=6,
    sanity=False,
    **kwargs
):

    print("transforms", transform)
    print("target transforms", target_transform)

    if 'multiclass' in kwargs.keys():
        print('multi class prediction dataloaders')
        if 'multiframe' in kwargs.keys(): 
            dataset_cls = ProjectionDatasetMultiClassSequence
        else: 
            dataset_cls = ProjectionDatasetMultiClass
        multiclass = kwargs['multiclass']
    else:
        dataset_cls = ProjectionDataset
        kwargs['multiclass'] = None 

    if transform is None:
        # resize_transform = ResizeImage((384, 512))
        resize_transform = None
    else:
        # resize_transform = ResizeImage((512, 512))
        resize_transform = transform

    if target_transform is None:
        # resize_label_transform = ResizeLabel((384, 512))
        resize_label_transform = None
    else:
        # resize_label_transform = ResizeLabel((512, 512))
        resize_label_transform = target_transform

    augmentation_instances = {}
    if augmentations is not None:
        augmentation_instances_train = get_augmentations(augmentations[0])
        augmentation_instances_valid = get_augmentations(augmentations[1])
        augmentation_instances_test = get_augmentations(augmentations[2])

    train_ds = dataset_cls(
        split_dict["train"],
        transform=resize_transform,
        target_transform=resize_label_transform,
        augmentations=augmentation_instances_train,
        prefix=prefix,
        sanity=sanity, 
        **kwargs,
    )
    valid_ds = dataset_cls(
        split_dict["valid"],
        transform=resize_transform,
        target_transform=resize_label_transform,
        prefix=prefix,
        sanity=sanity,
        augmentations=augmentation_instances_valid,
        **kwargs,
    )
    test_ds = dataset_cls(
        split_dict["test"],
        transform=resize_transform,
        target_transform=resize_label_transform,
        prefix=prefix,
        sanity=sanity,
        augmentations=augmentation_instances_test,
        **kwargs,
    )

    trainloader = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle_train, drop_last=True, pin_memory=True
    )
    validloader = DataLoader(
        valid_ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle_valid, drop_last=True, pin_memory=True
    )
    testloader = DataLoader(
        test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle_test, drop_last=True,
    )

    return trainloader, validloader, testloader


class ProjectionDataset(Dataset):
    def __init__(
        self,
        data_paths,
        transform=None,
        target_transform=None,
        prefix=None,
        sanity=False,
        augmentations=None,
        **kwargs,
    ):

        self.data = data_paths
        if sanity:
            print("sanity: truncating data")
            self.data = data_paths[:sanity]

        self.transform = transform
        self.target_transform = target_transform
        self.augmentations = augmentations if augmentations is not None else {}

        if prefix is not None:

            # if the path is complete and generated from local: truncate it
            if "/" in data_paths[0]:
                data_paths = [
                    "/".join(file_path.split("/")[-2:]) for file_path in data_paths
                ]

            # attach the prefix to the filepath
            self.data = [
                os.path.join(prefix, file_path).replace("\\", "/")
                for file_path in data_paths
            ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path_h5 = self.data[idx]
        projections, labels = h5_load(path_h5)
        if self.transform:
            projections = self.transform(projections)
        if self.target_transform:
            labels = self.target_transform(labels)

        projections = np.copy(np.flip(projections, axis=1))
        labels = np.copy(np.flip(labels, axis=1))

        for aug_name, augment_instance in self.augmentations.items():
            if "flip" in aug_name.lower() or "crop" in aug_name.lower():
                projections, labels = augment_instance(projections, labels)
            else:
                projections = augment_instance(projections)

        return projections, labels


class ProjectionDatasetMultiClass(Dataset):
    def __init__(
        self,
        data_paths,
        transform=None,
        target_transform=None,
        prefix=None,
        sanity=False,
        augmentations: Optional[dict] = None,
        multiclass: Optional[str] = None,
        **kwargs,
    ):

        self.data = data_paths
        if sanity:
            print("sanity: truncating data")
            self.data = [data_paths[0]]
        self.transform = transform
        self.target_transform = target_transform
        self.augmentations = augmentations if augmentations is not None else {}

        # set the multiclass label handling setting
        if multiclass not in ['stack', 'joint', 'collapsed', 'separate']:
            raise ValueError('multiclass setting %s not supported '
                             % (multiclass))
        self.multiclass = multiclass

        if prefix is not None:

            # if the path is complete and generated from local: truncate it
            if "/" in data_paths[0]:
                data_paths = [
                    "/".join(file_path.split("/")[-2:]) for file_path in data_paths
                ]

            # attach the prefix to the filepath
            self.data = [
                os.path.join(prefix, file_path).replace("\\", "/")
                for file_path in data_paths
            ]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path_h5 = self.data[idx]
        obj = h5_multi_load(path_h5)
        projections = obj["forward_vol"]
        tree_labels = obj["forward_vasc"]
        cath_labels = obj["forward_cath"]

        projections, tree_labels, cath_labels = self._correct_channel_dims(projections, tree_labels, cath_labels)

        if self.transform:
            projections = self.transform(projections)
        if self.target_transform:
            tree_labels = self.target_transform(tree_labels)
            cath_labels = self.target_transform(cath_labels)

        # projections = np.copy(np.flip(projections, axis=1))
        # tree_labels = np.copy(np.flip(tree_labels, axis=1))
        # cath_labels = np.copy(np.flip(cath_labels, axis=1))

        # apply augmentations
        
        for aug_name, augment_instance in self.augmentations.items():
            if "flip" in aug_name.lower() or 'crop' in aug_name.lower():
                projections, tree_labels, cath_labels = augment_instance(projections, tree_labels, cath_labels)
            else:
                projections = augment_instance(projections)
        
        if self.multiclass == 'stack':
            return projections.astype(np.float32), np.vstack([tree_labels, cath_labels]).astype(np.float32)

        elif self.multiclass == 'joint':
            return projections, np.where(tree_labels + cath_labels > 0, 1, 0).astype(np.int).squeeze()
        
        elif self.multiclass == 'collapsed': 
            return projections, np.where(cath_labels == 1, 2, tree_labels).astype(np.int).squeeze()

        elif self.multiclass == 'separate':
            return projections, tree_labels, cath_labels

        else:
            raise RuntimeError('oop multiclass switch failed')
        
    def _correct_channel_dims(self, *arrs: List[np.ndarray]) -> List[np.ndarray]: 
        """_correct_channel_dims

        For each of input arrays, insert a channel dimension at axis 0 if the
        current array is only of dimension 2 (infer that it is H, W). 

        Args:
            projections (List[np.ndarray]): np.ndarrays with dimension at least 2

        Returns:
            List[np.ndarray]: returned np.ndarrays with correct dimension
        """
        return_arr = []
        for arr in arrs: 
            if len(arr.shape) == 2: 
                return_arr.append(arr[None, :,:])
            elif len(arr.shape) < 2 :
                raise ValueError('received array of shape %s, unable to infer' % arr.shape) 
            else: 
                return_arr.append(arr)
        
        return return_arr


class SeqInferenceDataset(): 

    def __init__(self, 
                data_arr: np.ndarray, 
                angles: np.array, 
                labelp_vasc: Optional[np.ndarray] = None,
                labelp_cath: Optional[np.ndarray] = None,
                batch_size: Optional[int] = 1,
                nframes: Optional[int]=5, 
                target_spacing: Optional[int]=3, 
                augmentations: Optional[Dict[str, Callable]]=None,
                verbose=False) -> None:

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.data_arr = torch.from_numpy(data_arr).to(self.device)

        # store optional labelmaps for iteration as well 
        self.labelp_vasc = torch.from_numpy(labelp_vasc).to(self.device) if labelp_vasc is not None else labelp_vasc
        self.labelp_cath = torch.from_numpy(labelp_cath).to(self.device) if labelp_cath is not None else labelp_cath

        if self.labelp_vasc is not None and self.labelp_cath is not None: 
            self.has_gt = True
        else: 
            self.has_gt = False

        # store sequence information, batch information
        self.angles = torch.from_numpy(angles).to(self.device)
        self.target_spacing = target_spacing
        self.batch_size = batch_size
        self.nframes = nframes
        self.verbose = verbose

        # default augmentation is empty dict
        self.augmentations = augmentations if augmentations is not None else {}

        # infer angular direction: 
        self.rot_ang = torch.sign(torch.mean(self.angles[1:] - self.angles[:-1]))
        # we use this scalar such taht we can automatically infer from [0, 3, 6, 9, 12] -> [0, -3, -6, -9, -12]

    def infer_frames_from_geo(self, start: int): 
        # expect start to be an angular position?
        
        target_frame_angles = start + self.rot_ang * torch.arange(0, self.nframes, device=self.angles.device) * self.target_spacing
        ang_diff = torch.abs(self.angles[None, :] - target_frame_angles[:, None]) # result: shape[n_frames, all_angles]
        ang_idx = ang_diff.argmin(dim=1) # shape (n_frames, 1)
        ang_idx = ang_idx.clip(0, len(self.angles) - 1)
        
        # sanity check the result
        self._print(ang_idx, self.angles[ang_idx], target_frame_angles)
        
        return ang_idx
    
    def make_index_batches(self): 

        # fetch the corresponding indices that are closest to our target spacing between frames
        batch_indices = []
        for i in range(len(self.angles)): 
            batch_indices.append(self.infer_frames_from_geo(self.angles[i]))

        # stack the batch indices
        batch_indices = torch.stack(batch_indices, dim=0) # should be of shape (n_feasible angles, n_frames)
        batch_indices = batch_indices.to(self.device)
        
        # compute splits
        nb_total = batch_indices.shape[0]
        splits = [self.batch_size] * (nb_total // self.batch_size ) 
        if nb_total % self.batch_size > 0: 
            splits = splits + [nb_total % self.batch_size]

        self._print(splits, len(splits), nb_total // self.batch_size, nb_total % self.batch_size)
        batch_indices = torch.split(batch_indices.long(), splits, dim=0)

        return batch_indices
    
    def batch_generator(self): 

        batch_indices = self.make_index_batches()
        for batch in batch_indices:
            returns = [batch]
            return_arr = []
            for b_i in range(batch.shape[0]): 
                return_arr.append(self.data_arr[batch[b_i], :, :])  # shape (self.nframes, h, w)
            return_arr = torch.stack(return_arr, dim=0) # shape (self.nbatch, self.nframes, h, w)

            returns.append(return_arr)

            if self.has_gt: 
                stack_vasc = []
                stack_cath = []
                for b_i in range(batch.shape[0]): 
                    stack_vasc.append(self.labelp_vasc[batch[b_i], :, :])  # shape (self.nframes, h, w)
                    stack_cath.append(self.labelp_cath[batch[b_i], :, :])  # shape (self.nframes, h, w)

                stack_vasc = torch.stack(stack_vasc, dim=0) # shape (self.nbatch, self.nframes, h, w)
                stack_cath = torch.stack(stack_cath, dim=0) # shape (self.nbatch, self.nframes, h, w)

                returns.append(stack_vasc)
                returns.append(stack_cath)

            # apply augmnetations to return arrs that are not the angle array
            for aug_name, augment_instance in self.augmentations.items(): 
                if "flip" in aug_name.lower() or 'crop' in aug_name.lower():
                    returned = augment_instance(*returns[1:])
                    returns[1:] = returned
                else:
                    returns[1] = augment_instance(returns[1])
            
            for i in range(1, len(returns)): 
                returns[i] = returns[i][:,:,None,:,:]

            yield returns

        return 
    
    def _print(self, *args, **kwargs) -> None: 
        if self.verbose: 
            print(*args, **kwargs)
        return 

    

class ProjectionDatasetMultiClassSequence(Dataset):
    def __init__(
        self,
        data_paths,
        transform=None,
        target_transform=None,
        prefix=None,
        sanity=False,
        augmentations: Optional[dict] = None,
        multiclass: Optional[str] = None,
        multiframe: Optional[int] = None,
        framestep: Optional[int] = None,
        **kwargs,
    ):
        self.multiframe = 1 if multiframe is None else multiframe
        self.framestep = framestep if framestep else 1
        self.data = data_paths
        if sanity:
            print("sanity: truncating data")
            self.data = [data_paths[i] for i in range(2)]
        self.transform = transform
        self.target_transform = target_transform
        self.augmentations = augmentations if augmentations is not None else {}

        # set the multiclass label handling setting
        if multiclass not in ['stack', 'joint', 'collapsed', 'separate']:
            raise ValueError('multiclass setting %s not supported '
                             % (multiclass))
        self.multiclass = multiclass

        if prefix is not None:

            # if the path is complete and generated from local: truncate it
            if "/" in data_paths[0]:
                data_paths = [
                    "/".join(file_path.split("/")[-2:]) for file_path in data_paths
                ]

            # attach the prefix to the filepath
            self.data = [
                os.path.join(prefix, file_path).replace("\\", "/")
                for file_path in data_paths
            ]
    
    def infer_frames(self, idx: int) -> List[str]: 
        
        # get the current frame path 
        cur_frame_path = self.data[idx]

        path_chunks = [s for s in cur_frame_path.strip().split('/')]
        fname = path_chunks[-1]
        # remove the filetype suffix 
        fprefix, ftype = fname.split('.')
        fprefix_comps = fprefix.split('_')
        proj_position = int(fprefix_comps[-1])
        position_offsets = list(range(-self.multiframe // 2 + 1, self.multiframe // 2 + 1, self.framestep))
        proj_arc = [proj_position + i for i in position_offsets]

        all_proj_pos = [p % 120 for p in proj_arc]
        final_strs = [
            os.path.join('/'.join(path_chunks[:-1]), '%s_%d.%s' % ('_'.join(fprefix_comps[:-1]), proj_pos, ftype)) 
            for proj_pos in all_proj_pos
        ]
        return final_strs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        all_frame_paths = self.infer_frames(idx)
        projs = []
        tree_labels = []
        cath_labels = []
        for fp in all_frame_paths: 
            proj, tlab, clab = self._get_single_item(fp)
            projs.append(proj)
            tree_labels.append(tlab)
            cath_labels.append(clab)
            
        stacked_proj = np.stack(projs, axis=0).squeeze(1)
        tree_labels = np.stack(tree_labels, axis=0).squeeze(1)
        cath_labels = np.stack(cath_labels, axis=0).squeeze(1)
        
        # apply augmentations        
        for aug_name, augment_instance in self.augmentations.items():
            if "flip" in aug_name.lower() or 'crop' in aug_name.lower():
                [stacked_proj, 
                 tree_labels, cath_labels] = augment_instance(stacked_proj, 
                                                                tree_labels, 
                                                                cath_labels)
            else:
                stacked_proj = augment_instance(stacked_proj)
        
        stacked_labs = np.stack([tree_labels, cath_labels], axis=1)
        
        return stacked_proj[:, None, :, :].astype(np.float32), stacked_labs.astype(np.float32)

    def _get_single_item(self, path_h5: str) -> List[np.ndarray]:
        obj = h5_multi_load(path_h5)
        projections = obj["forward_vol"]
        tree_labels = obj["forward_vasc"]
        cath_labels = obj["forward_cath"]

        projections, tree_labels, cath_labels = self._correct_channel_dims(projections, tree_labels, cath_labels)

        if self.transform:
            projections = self.transform(projections)
        if self.target_transform:
            tree_labels = self.target_transform(tree_labels)
            cath_labels = self.target_transform(cath_labels)

        return projections, tree_labels, cath_labels
        
    def _correct_channel_dims(self, *arrs: List[np.ndarray]) -> List[np.ndarray]: 
        """_correct_channel_dims

        For each of input arrays, insert a channel dimension at axis 0 if the
        current array is only of dimension 2 (infer that it is H, W). 

        Args:
            projections (List[np.ndarray]): np.ndarrays with dimension at least 2

        Returns:
            List[np.ndarray]: returned np.ndarrays with correct dimension
        """
        return_arr = []
        for arr in arrs: 
            if len(arr.shape) == 2: 
                return_arr.append(arr[None, :,:])
            elif len(arr.shape) < 2 :
                raise ValueError('received array of shape %s, unable to infer' % arr.shape) 
            else: 
                return_arr.append(arr)
        
        return return_arr