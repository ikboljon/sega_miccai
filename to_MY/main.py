import os
import sys
import json
import shutil
import tempfile
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from monai.losses import DiceLoss, DiceCELoss, FocalLoss, GeneralizedDiceFocalLoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import (
    AsDiscrete,
    Activations,
    Compose,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR, UNet, SegResNet, UNETR
from monai import data
from monai.metrics import DiceMetric
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)
from functools import partial

import torch
import SimpleITK as sitk

from einops import rearrange


def main():
    """
    This is a brief description of the main function.
    """
    print("Hello, world!")

if __name__ == '__main__':
    main()