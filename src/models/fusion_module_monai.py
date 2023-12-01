from random import sample
from typing import Any, List
from einops import rearrange
from git import Tag
import pandas as pd
import numpy as np

import torch
from torch import nn

import torch
from lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, CyclicLR, CosineAnnealingLR

import torch.nn as nn
# import segmentation_models_pytorch as smp

from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss, GeneralizedDiceLoss, FocalLoss
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR, SwinUNETR, SegResNet
from monai.transforms import (
    AsDiscrete,
    Activations,
)
from monai import transforms
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.data import decollate_batch


from src.models.components.losses import Dice_and_FocalLoss #, DiceLoss, FocalLoss, BCELoss, Dice_and_BCELoss
from src.models.components.metrics import dice, recall, precision, dice_hecktor, compute_agg_dice, compute_volumes, get_intermediate_metrics
from src.models.components.models import BaselineUNet, FastSmoothSENormDeepUNet_supervision_skip_no_drop
from src.models.components.legonet import LegoNet
from src.models.components.unetr.unetr import CustomUNETR
from src.models.components.unest.unest import UNesT
from src.models.components.uxnet.uxnet import UXNET
from src.models.components.slim_unetr.slim_unetr import SlimUNETR
from src.models.components.meter import AverageMeter

from sklearn.metrics import roc_auc_score
from functools import partial

# torch.set_float32_matmul_precision('medium')

class FusionModule(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,

        **kwargs
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.infer_overlap = self.hparams['infer_overlap']
        self.sw_batch_size = self.hparams['sw_batch_size']



       
        if self.hparams['model'] == 'unet':
            self.model = BaselineUNet(in_channels=2, n_cls=3, n_filters=24)

        elif self.hparams['model'] == 'senet':
            self.model = FastSmoothSENormDeepUNet_supervision_skip_no_drop(
                            in_channels=2, n_cls=3, n_filters=24, reduction=2)

        elif self.hparams['model'] == 'segresnet':
            self.model = SegResNet(
                            in_channels=2, 
                            out_channels=3, 
                            init_filters=32,
                            norm='BATCH', 
                            blocks_down=(1,2,2,4,4,4), 
                            blocks_up=(1,1,1,1,1), 
                            upsample_mode='deconv',)

        elif self.hparams['model'] == 'unetr':
            self.model = UNETR(in_channels=2, 
                               out_channels=3, 
                               feature_size=16,
                               img_size=self.hparams['roi'])
            
        elif self.hparams['model'] == 'swinunetr':
            self.model = SwinUNETR(
                                    img_size=self.hparams['roi'],
                                    in_channels=2,
                                    out_channels=3,
                                    feature_size=24,
                                    drop_rate=0.0,
                                    attn_drop_rate=0.0,
                                    dropout_path_rate=0.0,
                                    use_checkpoint=True,
                                    use_v2=True,
                                    )

        elif self.hparams['model'] == 'custom_unetr':
            self.model = CustomUNETR(in_channels=2, 
                               out_channels=3, 
                               feature_size=16,
                               img_size=self.hparams['roi'],
                               )
            
        elif self.hparams['model'] == 'legonet':
            self.model = LegoNet(
                                in_channels=2,
                                out_channels=3,
                                depths=[2, 2, 2, 2],
                                feature_size=[24,48,96,192], #[48, 96, 192, 384], 
                                drop_path_rate=0,
                                layer_scale_init_value=1e-6,
                                spatial_dims=3,
                                hidden_size=768,
                                return_logits = False,
                                
                            )
            
        elif self.hparams['model'] == 'unest':
            self.model = UNesT(in_channels=2,
                                out_channels=3,
                                # feature_size=24,
                                img_size=self.hparams['roi']
                            )
            
        elif self.hparams['model'] == 'uxnet':
            self.model = UXNET(
                                in_chans=2,
                                out_chans=3,
                                depths=[2, 2, 2, 2],
                                feat_size=[24,48,96,192], #[48, 96, 192, 384],
                                drop_path_rate=0,
                                layer_scale_init_value=1e-6,
                                spatial_dims=3,
                            )
            
        elif self.hparams['model'] == 'slim_unetr':
            self.model = SlimUNETR(
                                in_channels=2,
                                out_channels=3,
                                embed_dim=96,
                                embedding_dim=216,
                                channels=(24, 48, 60),
                                blocks=(1, 2, 3, 2),
                                heads=(1, 2, 4, 4),
                                r=(4, 2, 2, 1),
                                dropout=0.3,
                            )
            



        # elif self.hparams['model'] == 'swin':
        #     self.model = SwinUNETR(
        #                             img_size=(128,128,128),
        #                             in_channels=1,
        #                             out_channels=14,
        #                             feature_size=48)

        #     checkpoint = torch.load("/home/ikboljon.sobirov/data/shared/ikboljon.sobirov/rima/lightning-hydra-template/weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt")
        #     state_dict = checkpoint['state_dict']
        #     for key in list(state_dict):
        #         state_dict[key.replace("module.", "swinViT.")] = state_dict.pop(key)

        #     self.model.load_state_dict(state_dict)
        #     self.model.out = nn.Conv3d(48, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))

        else:
            print('Please select the correct model architecture name.')
        
        torch.backends.cudnn.benchmark = True
        self.dice_loss = DiceCELoss(to_onehot_y=True, softmax=False)
        self.post_sigmoid = Activations(softmax=True)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=3)
        self.post_label = AsDiscrete(to_onehot=3)


        self.validation_step_loss = []
        self.intermediate_metrics = []

        self.run_loss_val = AverageMeter()
        self.run_loss_train = AverageMeter()


    def forward(self, x: torch.Tensor):
        return self.model(x)

    def init_params(self, m: torch.nn.Module):
        """Initialize the parameters of a module.
        Parameters
        ----------
        m
            The module to initialize.
        Notes
        -----
        Convolutional layer weights are initialized from a normal distribution
        as described in [1]_ in `fan_in` mode. The final layer bias is
        initialized so that the expected predicted probability accounts for
        the class imbalance at initialization.
        References
        ----------
        .. [1] K. He et al. ‘Delving Deep into Rectifiers: Surpassing
           Human-Level Performance on ImageNet Classification’,
           arXiv:1502.01852 [cs], Feb. 2015.
        """

        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, a=.1)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            # initialize the final bias so that the predictied probability at
            # init is equal to the proportion of positive samples
            nn.init.constant_(m.bias, -1.5214691)


    def step(self, batch: Any):
        
        sample = batch

        pred_mask = self.forward(sample['ctpt'])

        loss = self.dice_loss(pred_mask, sample['seg'])

        return loss

    def training_step(self, batch: Any, batch_idx: int):
        
        loss = self.step(batch)
    
        # log train metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss}

    def on_train_epoch_end(self):

        self.run_loss_train.reset()


        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def val_step(self, batch: Any):
        
        sample = batch

        pred_mask = self.forward(sample['ctpt'])

        loss = self.dice_loss(pred_mask, sample['seg'])

        return loss, pred_mask
    
    def validation_step(self, batch: Any, batch_idx: int):
        
        input, target = batch['ctpt'], batch['seg']

        logits = sliding_window_inference(input, self.hparams['roi'], self.sw_batch_size, self.model, self.infer_overlap)
        loss = self.dice_loss(logits, target)


        self.validation_step_loss.append(loss.item())

        val_labels_list = decollate_batch(target)
        val_labels_convert = [self.post_label(val_label_tensor) for val_label_tensor in val_labels_list]
        
        val_outputs_list = decollate_batch(logits)

        val_output_convert = [self.post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
        
        merged_channels_pred = torch.argmax(val_output_convert[0], dim=0)
        merged_channels_label = torch.argmax(val_labels_convert[0], dim=0)
        self.intermediate_metrics.append(get_intermediate_metrics(merged_channels_label, merged_channels_pred))

        
        self.run_loss_val.update(loss.item(), n=self.hparams['batch_size'])

        
        return {"loss": torch.from_numpy(self.run_loss_val.avg)}
        # pass

    def on_validation_epoch_end(self):

        self.run_loss_val.reset()

        hecktor_metrics = compute_agg_dice(self.intermediate_metrics)

        loss        = np.stack(self.validation_step_loss).mean()

        log = {
               "val/loss": loss,
                "val/dice": hecktor_metrics['AggregatedDsc']['mean'],
                "val/dice_tum": hecktor_metrics['AggregatedDsc']['GTVp'],
                "val/dice_nod": hecktor_metrics['AggregatedDsc']['GTVn'],
               }
        
        self.log_dict(log)


        self.validation_step_loss.clear()
        self.intermediate_metrics.clear()
        

        return {"val/loss": loss,
                "val/dice": hecktor_metrics['AggregatedDsc']['mean'],
                "val/dice_tum": hecktor_metrics['AggregatedDsc']['GTVp'],
                "val/dice_nod": hecktor_metrics['AggregatedDsc']['GTVn'],
               }

        # pass

    # def test_step(self, batch: Any, batch_idx: int):
        

    #     # return self.validation_step(batch, batch_idx)
    #     pass

    # def test_epoch_end(self, outputs: List[Any]):


    #     pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        # optimizer = make_optimizer(AdamW, self.model, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        
        optimizer = AdamW(self.parameters(),
                         lr=self.hparams.lr,
                         weight_decay=self.hparams.weight_decay)
        scheduler = {
            "scheduler": CosineAnnealingLR(optimizer, T_max=self.hparams['max_epochs'], eta_min=1e-5),
            # "scheduler": CosineAnnealingWarmRestarts(optimizer, T_0=25, eta_min=1e-5),
            # "scheduler": MultiStepLR(optimizer, milestones=[75], gamma=0.05),
            #"scheduler": CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-2),
            # "scheduler": ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True),
            "monitor": "val/loss",
        }
        return [optimizer], [scheduler]
    