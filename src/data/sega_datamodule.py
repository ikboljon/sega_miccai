from typing import Optional, Tuple
from sklearn.model_selection import KFold, StratifiedKFold

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, Subset
from src.data.components.fusion_dataset import SegaDataset
from src.data.augmentations import *
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

        # self.transforms = tio.transforms.Compose([
        #                             # tio.transforms.ToTensor(),
        #                             tio.transforms.Resize((144, 144, 144)),
        #                             # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    # tio.transforms.ZNormalization(),
        # ])
        self.train_transforms = Compose([
                                # RandomRotation(p=0.5, angle_range=[0, 30]),
                                # Mirroring(p=0.5),
                                # AdjustContrast(gamma=1),
                                # ElasticDeformation(),
                                NormalizeIntensity(),
                                # ToTensor(), 
                                # Resizing(z=256,x=256,y=256),
                            ])

        self.val_transforms = Compose([
                                NormalizeIntensity(),
                                # ToTensor(), 
                                # Resizing(z=256,x=256,y=256),
                            ])                    
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
        
        self.dataset = SegaDataset(self.hparams["data_dir"],
                                      transforms=self.train_transforms, 
                                      )
        
        full_indices = range(len(self.dataset))

        kf = KFold(n_splits=5, shuffle=True, random_state=786)

        train_idx = {}
        test_idx = {}

        key = 1
        for i,j in kf.split(full_indices):
            train_idx[key] = i
            test_idx[key] = j

            key += 1

        train_dataset, val_dataset = Subset(self.dataset, train_idx[self.Fold]), Subset(self.dataset, test_idx[self.Fold])
        val_dataset.dataset.transform = self.val_transforms
        print(len(self.dataset), len(train_dataset), len(val_dataset))

        self.data_train = train_dataset
        self.data_val = val_dataset
        
        data = next(iter(self.data_train))
        print(data['input'].shape)
        print(data['target'].shape)
        print(data['id'])


    def train_dataloader(self):
        return DataLoader(
                dataset=self.data_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=True,
                drop_last=True,
            )
    
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=True,
        )