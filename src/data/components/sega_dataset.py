import torch
import os
import nrrd
import matplotlib.pyplot as plt
import pathlib
from torch.utils.data import Dataset, DataLoader
import numpy as np
import SimpleITK as sitk
from einops import  rearrange
import nibabel as nib


def get_patient_files(path_to_imgs):

    path_to_imgs = pathlib.Path(path_to_imgs)

    patients = [p for p in os.listdir(path_to_imgs) if os.path.isdir(path_to_imgs / p)]

    paths = []

    for p in patients:
        path_to_ct = path_to_imgs / p / (p + '_ct.nii.gz')
        path_to_gt = path_to_imgs / p / (p + '_gt.nii.gz')

        paths.append((path_to_ct, path_to_gt))
    return paths

class SegaDataset(Dataset):
    
    def __init__(self, paths_to_samples, transforms=None):
        self.transforms = transforms
        self.paths = get_patient_files(paths_to_samples)


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
       
        sample = dict()

        id_ = self.paths[index][0].parent.stem

        sample['id']  = id_
        img_data = self.read_nifti(self.paths[index][0])
        # img_data = rearrange(img_data, 'h w d -> d w h')
        # img_data = sitk.GetArrayFromImage(img)
        img_data = np.expand_dims(img_data, axis=3)
        sample['input'] = img_data

        mask_data = self.read_nifti(self.paths[index][-1])
        # mask_data = rearrange(mask_data, 'h w d -> d w h')
        # mask_data = sitk.GetArrayFromImage(mask)
        mask_data = np.expand_dims(mask_data, axis=3)

        sample['target'] = mask_data

        if self.transforms:
            sample = self.transforms(sample)
        
        return sample


    @staticmethod
    def read_nrrd_file(path):
        img = sitk.ReadImage(str(path), sitk.sitkFloat32)

        return img
        
    @staticmethod
    def read_nifti(path_to_nifti, return_numpy=True):
        """Read a NIfTI image. Return a numpy array (default) or `nibabel.nifti1.Nifti1Image` object"""
        if return_numpy:
            return nib.load(str(path_to_nifti)).get_fdata()
        return nib.load(str(path_to_nifti))

    @staticmethod
    def load_nrrd(full_path_filename,dtype=sitk.sitkFloat32):
        '''
        N*h*W
        :param full_path_filename:
        :return:*H*W
        '''
        if not os.path.exists(full_path_filename):
            raise FileNotFoundError
        image = sitk.ReadImage(full_path_filename)
        image = sitk.Cast(sitk.RescaleIntensity(image), dtype)
        # data = sitk.GetArrayFromImage(image) # N*H*W
        return image