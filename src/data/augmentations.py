import random
import numpy as np
import torch
import elasticdeform
import sys

import torchio as tio 
from skimage.transform import rotate
import pathlib
import os
from monai.transforms.transform import Transform
from monai.config.type_definitions import NdarrayOrTensor, NdarrayTensor
from monai.utils.enums import TransformBackends
from monai.utils.module import look_up_option
from monai.transforms.croppad.array import Pad
from monai.transforms import Resized
from monai.utils import (
    InterpolateMode,
    NumpyPadMode,
    PytorchPadMode,
    ensure_tuple_rep
)
from typing import Any, List, Optional, Sequence, Tuple, Union
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type
import SimpleITK as sitk

class Compose:
    def __init__(self, transforms=None):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)

        return sample


class LoadImagedMonai:
    def __init__(self, keys=["image", "image2", "label"], ensure_channel_first = True):
        self.keys=keys
        self.chnl_first=ensure_channel_first
        # self.path_to_data_dir=path_to_data_dir

    def __call__(self, path_to_data_dir):
        if path_to_data_dir is None:
            print('Please provide directory to the data path')
        else:
            sample = dict()
            # sample['id']  = 
            # paths = self.get_patient_files(path_to_data_dir)
            ct_data = self.read_nii_file(path_to_data_dir[self.keys[0]])
            pt_data = self.read_nii_file(path_to_data_dir[self.keys[1]])
            gt_data = self.read_nii_file(path_to_data_dir[self.keys[2]])
            
            # gt_tum = gt_data.copy()
            # gt_tum[gt_tum == 2] = 0

            # gt_nod = gt_data.copy()
            # gt_nod[gt_nod == 1] = 0
            # # gt_nod[gt_nod == 2] = 1

            # gt_new = np.stack([gt_nod, gt_tum], axis=0)
            
            ct_data = torch.from_numpy(ct_data).float().unsqueeze(0)
            pt_data = torch.from_numpy(pt_data).float().unsqueeze(0)
            gt_data = torch.from_numpy(gt_data).float().unsqueeze(0)
            
            sample['image'] = torch.concat([ct_data, pt_data], dim=0)
            sample['label'] = gt_data
            sample['fold'] = path_to_data_dir['fold']
            sample['id'] = path_to_data_dir['id']

            return sample


    
    @staticmethod
    def get_patient_files(path_to_imgs):

        path_to_imgs = pathlib.Path(path_to_imgs)

        patients = [p for p in os.listdir(path_to_imgs) if os.path.isdir(path_to_imgs / p)]

        paths = []

        for p in patients:
            path_to_ct = path_to_imgs / p / (p + '_ct.nii.gz')
            path_to_pt = path_to_imgs / p / (p + '_pt.nii.gz')
            path_to_gt = path_to_imgs / p / (p + '_gt.nii.gz')

            paths.append((path_to_ct, path_to_pt, path_to_gt))
        return paths
            
    @staticmethod
    def read_torch_file(path):
        img = torch.load(path)

        return img
    
    @staticmethod
    def read_nii_file(path):
        img = sitk.GetArrayFromImage(sitk.ReadImage(path))

        return img


class Resizing:
    def __init__(self, mode='train', z=196, x=196, y=196):
        if mode not in ['train', 'test']:
            raise ValueError(f"Argument 'mode' must be 'train' or 'test'. Received {mode}")

        self.mode = mode
        self.z = z
        self.x = x
        self.y = y

    def __call__(self, sample):
        if self.mode == 'train':
            img, mask = sample['input'], sample['target']
            img = tio.Resize((self.z, self.x, self.y), image_interpolation='bspline')(img)
            mask = tio.Resize((self.z, self.x, self.y), image_interpolation='nearest')(mask)
            mask = torch.where(mask == 1., 1., 0.)
            sample['input'], sample['target'] = img, mask
        else:
            img = sample['input']
            img = tio.Resize((self.z, self.x, self.y), image_interpolation='bspline')(img)
            sample['input'] = img
        return sample 

class ToTensor:
    def __init__(self, mode='train'):
        if mode not in ['train', 'test']:
            raise ValueError(f"Argument 'mode' must be 'train' or 'test'. Received {mode}")
        self.mode = mode

    def __call__(self, sample):
        if self.mode == 'train':
            img, mask = sample['input'], sample['target']
            img = np.transpose(img, axes=[3, 0, 1, 2])
            mask = np.transpose(mask, axes=[3, 0, 1, 2])
            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).float()

            sample['input'], sample['target'] = img, mask

        else:  # if self.mode == 'test'
            img = sample['input']
            img = np.transpose(img, axes=[3, 0, 1, 2])
            img = torch.from_numpy(img).float()
            sample['input'] = img

        return sample        


class NormalizeIntensity:

    def __call__(self, sample):
        img = sample['input']
        img = self.normalize_ct(img)

        sample['input'] = img
        return sample

    @staticmethod
    def normalize_ct(img):
        norm_img = np.clip(img, -1024, 1024) / 1024
        return norm_img


class Mirroring:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            img, mask = sample['input'], sample['target']

            n_axes = random.randint(0, 3)
            random_axes = random.sample(range(3), n_axes)


            img = np.flip(img, axis=tuple(random_axes))
            mask = np.flip(mask, axis=tuple(random_axes))

            sample['input'], sample['target'] = img.copy(), mask.copy()

        return sample

class RandomRotation:

    def __init__(self, p=0.5, angle_range=[5, 15]):
        self.p = p
        self.angle_range = angle_range

    def __call__(self, sample):
        if random.random() < self.p:
            img, mask = sample['input'], sample['target']

            num_of_seqs = img.shape[-1]
            n_axes = random.randint(1, 3)
            random_axes = random.sample([0, 1, 2], n_axes)

            for axis in random_axes:

                angle = random.randrange(*self.angle_range)
                angle = -angle if random.random() < 0.5 else angle

                # for i in range(num_of_seqs):
                img[:, :, :, 0] = RandomRotation.rotate_3d_along_axis(img[:, :, :, 0], angle, axis, 1)

                mask[:, :, :, 0] = RandomRotation.rotate_3d_along_axis(mask[:, :, :, 0], angle, axis, 0)

            sample['input'], sample['target'] = img, mask
        return sample

    @staticmethod
    def rotate_3d_along_axis(img, angle, axis, order):

        if axis == 0:
            rot_img = rotate(img, angle, order=order, preserve_range=True)

        if axis == 1:
            rot_img = np.transpose(img, axes=(1, 2, 0))
            rot_img = rotate(rot_img, angle, order=order, preserve_range=True)
            rot_img = np.transpose(rot_img, axes=(2, 0, 1))

        if axis == 2:
            rot_img = np.transpose(img, axes=(2, 0, 1))
            rot_img = rotate(rot_img, angle, order=order, preserve_range=True)
            rot_img = np.transpose(rot_img, axes=(1, 2, 0))

        return rot_img

class AdjustContrast(Transform):
    """
    Changes image intensity by gamma. Each pixel/voxel intensity is updated as::
        x = ((x - min) / intensity_range) ^ gamma * intensity_range + min
    Args:
        gamma: gamma value to adjust the contrast as function.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, gamma: float, random=True) -> None:
        if not isinstance(gamma, (int, float)):
            raise ValueError("gamma must be a float or int number.")
        self.gamma = gamma
        self.random = random

    def __call__(self, sample):
        """
        Apply the transform to `img`.
        """
        if self.random:
            self.gamma = np.random.uniform(0.5, 2.0)

        images, mask = sample['input'], sample['target']
        ct_img = images[:,:,:,0]
        pet_img = images[:,:,:,1]
        
        
        epsilon = 1e-7
        img_min = pet_img.min()
        img_range = pet_img.max() - img_min
        
        ret: NdarrayOrTensor = ((pet_img - img_min) / float(img_range + epsilon)) ** self.gamma * img_range + img_min
        img = np.stack([ct_img, ret], axis=-1)

        sample['input'] = img

        return sample

class Zoom(Transform):
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, sample):
        images, mask = sample['input'], sample['target']
        ct_img = images[:,:,:,0]
        pet = images[:,:,:,1]
        mask = mask.squeeze(-1)
        
        zoomed_ct = zoom(ct_img, self.factor)
        zoomed_pet = zoom(pet, self.factor)
        
        img = np.stack([zoomed_ct, zoomed_pet], axis=-1)
        sample['input'] = img


        zoomed_mask = zoom(mask, self.factor)
        zoomed_mask[zoomed_mask<0.5] = 0
        zoomed_mask[zoomed_mask>0] = 1  
        sample['target'] = np.expand_dims(zoomed_mask, axis=-1)

        return sample


def zoom(
    img,
    factor,
    padding_mode: Optional[Union[NumpyPadMode, PytorchPadMode, str]] = None,
    align_corners: Optional[bool] = True,
    keep_size = True,

) -> NdarrayOrTensor:
    """
    Args:
        img: channel first array, must have shape: (num_channels, H[, W, ..., ]).
        mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``self.mode``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
        padding_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            The mode to pad data after zooming.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Defaults to ``self.align_corners``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
    """
    
    img_t: torch.Tensor
    img_t, *_ = convert_data_type(img, torch.Tensor, dtype=torch.float32)  # type: ignore
    mode = InterpolateMode('bilinear')
    _zoom = ensure_tuple_rep(factor, img.ndim - 1)  # match the spatial image dim
    zoomed: NdarrayOrTensor = torch.nn.functional.interpolate(  # type: ignore
        recompute_scale_factor=True,
        input=img_t.unsqueeze(0),
        scale_factor=list(_zoom),
        mode=look_up_option(mode if mode is None else mode, InterpolateMode).value,
        align_corners=align_corners if align_corners is None else align_corners,
    )
    zoomed = zoomed.squeeze(0)

    if keep_size and not np.allclose(img_t.shape, zoomed.shape):

        pad_vec = [(0, 0)] * len(img_t.shape)
        slice_vec = [slice(None)] * len(img_t.shape)
        for idx, (od, zd) in enumerate(zip(img_t.shape, zoomed.shape)):
            diff = od - zd
            half = abs(diff) // 2
            if diff > 0:  # need padding
                pad_vec[idx] = (half, diff - half)
            elif diff < 0:  # need slicing
                slice_vec[idx] = slice(half, half + od)

        padder = Pad(pad_vec, padding_mode or padding_mode)
        zoomed = padder(zoomed)
        zoomed = zoomed[tuple(slice_vec)]

    out, *_ = convert_to_dst_type(zoomed, dst=img)
    return out



class ElasticDeformation():
    def __init__(self, p = 0.5):
        self.p = p
    def __call__(self, sample):
        images, mask = sample['input'], sample['target']
        ct_img = images[:,:,:,0]
        pet_img = images[:,:,:,1]
        if random.random()<self.p:
            new_ct, new_pet, new_mask = elasticdeform.deform_random_grid([ct_img,pet_img,mask],sigma = random.randint(5, 10), points =  random.randint(1,3),axis=(0, 1, 2))
            new_mask = (new_mask - np.min(new_mask))/(np.max(new_mask) - np.min(new_mask))
            new_mask[new_mask<0.5] = 0
            new_mask[new_mask>0] = 1
            img = np.stack([new_ct, new_pet], axis=-1)
            sample['input'], sample['target'] =img,new_mask
        return sample