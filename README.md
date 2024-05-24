# Vessel-Targeted Motion Compensation 

Source codes for network training associated with paper titled "Vessel-Targeted Compensation of Deformable Motion in Interventional Cone-Beam CT." 

### Dependencies
This code was built with `Pytorch` 1.13.0, `scikit-image` 0.19.2, `scipy` 1.9.3, `numpy` 1.22.3, `nibabel` 4.0.2.

Additionally, please see: 
[centerline DICE implementation](https://github.com/jocpae/clDice)


### Network training and running code

There are two main entry points to the provided code, for multi-view U-Net training and single-view U-Net training, respectively. 

**Multi-View U-Net training**: 

Example script call: 
```
python -u src/train_seq_unet.py --multiframe 5 --framestep 1 --batch_size 16 --epochs 15 --distributed
```

The arguments to the script `src/train_seq_unet.py` include the batch size (`--batch_size`), the total number of epochs (`--epochs`), the multi-view settings (number of frames: `--multiframe`, interval between frames: `--framestep`). Specifying `--distributed` casts the model using torch's `nn.DataParallel()`, which enables handling a larger batch size. 



**Single-View U-Net training**: 

Example script call: 
```
python -u src/train_unet.py
```

**Additional training details**:

For both single-view and multi-view U-Net training, additioanl configurations can be found in `src/train_unet.py` and `src/train_seq_unet.py`. These include: 
* training configurations (model architecture configuration; optimizer settings; loss weights) 
* log settings (checkpoint save location, giving a name to the training run, etc.)
* `split_dict_path`, which is expected to be the path to a numpy `.npz` file, which contains the training / validation / test split data. That is, `split_dict[k]` for `k in ['train', 'valid', 'test']` should be lists of filepaths corresponding to the data in the training, validation, and test splits of the dataset. 

The current dataloader classes expect forward projection data and labels stored as `.hdf5` files. Details can be inspected under `src/networks/dataloaders/sequence_loader.py` in the `__getitem__()` methods of the subclasses of `torch.utils.data.Dataset`. These methods should be re-implemented for custom training with a different dataset storage format. 

Example training data can be found in [example_data/](example_data/).

Trained network weights can be found in [Release v0.1_checkpoints](https://github.com/I-STAR/VesselTargetedMoCo/releases/tag/v0.1_checkpoints).