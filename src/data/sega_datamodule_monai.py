from typing import Optional, Tuple
from sklearn.model_selection import KFold, StratifiedKFold

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, Subset
from src.data.components.sega_dataset import SegaDataset
from src.data.augmentations import *
import monai
from monai import transforms
from monai import data

import json
import os
# from src.datamodules.transforms import *


class SegaDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters()

        # data transformations
        self.data_dir = self.hparams["data_dir"]
        #self.train_val_test_split = self.hparams["train_val_test_split"]
        self.batch_size = self.hparams["batch_size"]
        self.num_workers = self.hparams["num_workers"]
        self.pin_memory = self.hparams["pin_memory"]
        self.Fold = self.hparams["Fold"]
        self.json_dir = self.hparams["json_dir"]
        self.roi = self.hparams["roi"]



        self.train_transform = transforms.Compose(
        [
            # transforms.LoadImaged(keys=["image", "label"], ensure_channel_first = True, image_only=True),
            LoadImagedMonai(keys=["image", "label"], ensure_channel_first = True),
            # transforms.SpatialPadd(keys=["image", "label"], spatial_size=(176,176,176), method='end'),
            # transforms.Orientationd(keys=["image", "label"], axcodes="PLS"),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(self.roi[0], self.roi[1], self.roi[2]),
                pos=1,
                neg=1,
                num_samples=6,
                image_key="image",
                image_threshold=0,
            ),
            # transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            # # ClipCT(keys=["image"]),

            # transforms.RandFlipd(
            #     keys=["image", "label"],
            #     spatial_axis=[0],
            #     prob=0.20,
            # ),
            # transforms.RandFlipd(
            #     keys=["image", "label"],
            #     spatial_axis=[1],
            #     prob=0.20,
            # ),
            # transforms.RandFlipd(
            #     keys=["image", "label"],
            #     spatial_axis=[2],
            #     prob=0.20,
            # ),
            # transforms.RandRotate90d(
            #     keys=["image", "label"],
            #     prob=0.20,
            #     max_k=3,
            # ),
            
    
            # transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            # transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            # transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )

        self.val_transform = transforms.Compose(
        [
            # transforms.LoadImaged(keys=["image", "label"], ensure_channel_first = True, image_only=True),
            LoadImagedMonai(keys=["image", "label"], ensure_channel_first = True),
            # transforms.Orientationd(keys=["image", "label"], axcodes="PLS"),
            # transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )

        # self.transforms = tio.transforms.Compose([
        #                             # tio.transforms.ToTensor(),
        #                             tio.transforms.Resize((144, 144, 144)),
        #                             # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    # tio.transforms.ZNormalization(),
        # ])
        # self.train_transforms = Compose([
        #                         # RandomRotation(p=0.5, angle_range=[0, 30]),
        #                         # Mirroring(p=0.5),
        #                         # AdjustContrast(gamma=1),
        #                         # ElasticDeformation(),
        #                         NormalizeIntensity(),
        #                         # ToTensor(), 
        #                         # Resizing(z=256,x=256,y=256),
        #                     ])

        # self.val_transforms = Compose([
        #                         NormalizeIntensity(),
        #                         # ToTensor(), 
        #                         # Resizing(z=256,x=256,y=256),
        #                     ])                    
        # self.test_transforms = transforms.Compose([
        #                        NormalizeIntensity(), 
        #                        ToTensor()
        #                     ])
        
        # self.train_transform_2 = Compose([
        #             Mirroring(p=0.5),
        #             RandomRotation(p=0.5, angle_range=[0, 45]),
        #             NormalizeIntensity(),
        #             ToTensor()
        #         ])



        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None 


    def setup(self, stage: Optional[str] = None):
        
        # self.dataset = RimaDataset(self.hparams["data_dir"],
        #                               transforms=self.train_transforms, 
        #                               mode='train')
        
        # self.dataset = SegaDataset(self.hparams["data_dir"],
        #                               transforms=self.train_transform, 
        #                               )
        
        train_files, validation_files = datafold_read(datalist=self.json_dir, basedir=self.data_dir, fold=self.Fold)

        train_ds = data.Dataset(data=train_files, transform=self.train_transform)

        # train_loader = data.DataLoader(
        #     train_ds,
        #     batch_size=self.batch_size,
        #     shuffle=True,
        #     num_workers=8,
        #     pin_memory=True,
        # )

        val_ds = data.Dataset(data=validation_files, transform=self.val_transform)
        # val_loader = data.DataLoader(
        #     val_ds,
        #     batch_size=self.batch_size,
        #     shuffle=False,
        #     num_workers=8,
        #     pin_memory=True,
        # )

        # full_indices = range(len(self.dataset))

        # kf = KFold(n_splits=5, shuffle=True, random_state=786)

        # train_idx = {}
        # test_idx = {}

        # key = 1
        # for i,j in kf.split(full_indices):
        #     train_idx[key] = i
        #     test_idx[key] = j

        #     key += 1

        # train_dataset, val_dataset = Subset(self.dataset, train_idx[self.Fold]), Subset(self.dataset, test_idx[self.Fold])
        # val_dataset.dataset.transform = self.val_transforms
        # print(len(self.dataset), len(train_dataset), len(val_dataset))

        self.data_train = train_ds
        self.data_val = val_ds
        
        a = next(iter(self.data_train))
        # print(a['input'].shape)
        # print(a['target'].shape)
        # print(a['id'])


    def train_dataloader(self):
        train_loader = data.DataLoader(
                dataset=self.data_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=True,
                drop_last=True,
            )
        return train_loader
    
    
    def val_dataloader(self):
        val_loader = data.DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=True,
        )
        return val_loader
    
def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val