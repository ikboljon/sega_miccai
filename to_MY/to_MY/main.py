import os

from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
)

from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR, UNet, SegResNet, UNETR
from monai.metrics import DiceMetric
from monai.data import (
    decollate_batch,
)

# import the libraries
import torch
import yaml
import pathlib
import argparse
from utils import load_model_weights, AverageMeter
from loader import get_loader
from einops import rearrange
import SimpleITK as sitk
import pandas as pd

def main(args):
    """
    This is a brief description of the main function.
    """
    # read yaml file
    path_to_config = pathlib.Path(args.path)
    with open(path_to_config) as f:
        config = yaml.safe_load(f)

    path_to_data = pathlib.Path(config['path_to_data'])
    path_to_json = pathlib.Path(config['path_to_json'])
    path_to_save = pathlib.Path(config['path_to_save'])
    path_to_chkpt = pathlib.Path(config['path_to_chkpt'])
    
    batch_size = int(config['batch_size'])
    sw_batch_size = int(config['sw_batch_size'])
    fold = int(config['fold'])
    infer_overlap = float(config['infer_overlap'])
    roi = int(config['roi'])
    roi = tuple([roi]*3)

    val_loader, _ = get_loader(batch_size, path_to_data, path_to_json, fold, roi)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SegResNet(
                    in_channels=2, 
                    out_channels=3, 
                    init_filters=32,
                    norm='BATCH', 
                    blocks_down=(1,2,2,4,4,4), 
                    blocks_up=(1,1,1,1,1), 
                    upsample_mode='deconv',)


    model = load_model_weights(model, path_to_chkpt)

    model = model.to(device)

    post_pred = AsDiscrete(argmax=True, to_onehot=3)
    post_label = AsDiscrete(to_onehot=3)

    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")

    metric_dictionary = {}
    metric_dictionary['id'] = []
    metric_dictionary['dice'] = []
    metric_dictionary['dice_t'] = []
    metric_dictionary['dice_n'] = []

    model.eval()
    
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save, exist_ok=True)


    with torch.no_grad():
        for idx, sample in enumerate(val_loader):

            pat_id = sample['id'][0].split('/')[-1]
            
            val_inputs, val_labels = sample["ctpt"].to(device), sample["seg"].to(device)

            logits = sliding_window_inference(val_inputs, roi, sw_batch_size, model, infer_overlap)
            
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            
            val_outputs_list = decollate_batch(logits)

            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]



            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            dice_metric_batch(y_pred=val_output_convert, y=val_labels_convert)

            mean_dice_val = dice_metric.aggregate().item()
            metric_batch_val = dice_metric_batch.aggregate()

            metric_tumor = metric_batch_val[0].item()
            metric_lymph = metric_batch_val[1].item()

            dice_metric.reset()
            dice_metric_batch.reset()

            # break

            
            
            pat_id = sample['id'][0].split('/')[-1]
            print("Processing: ", pat_id, " Dice: ", mean_dice_val)

            metric_dictionary['id'].append(pat_id)
            metric_dictionary['dice'].append(mean_dice_val)
            metric_dictionary['dice_t'].append(metric_tumor)
            metric_dictionary['dice_n'].append(metric_lymph)


            merged_channels_pred = torch.argmax(val_output_convert[0], dim=0)
            sitk_pred = merged_channels_pred.cpu().numpy()
            # merged_channels_label = torch.argmax(val_labels_convert[0], dim=0)
            # sitk_label = merged_channels_label.cpu().numpy()

            # pred_list.append(sitk_pred)
            # label_list.append(sitk_label)
            sitk_pred = rearrange(sitk_pred, 'h w d -> d h w')
            
            sitk_pred = sitk.GetImageFromArray(sitk_pred)
            
            if not os.path.exists(os.path.join(path_to_save, pat_id)):
                        os.makedirs(os.path.join(path_to_save, pat_id), exist_ok=True)
            
            df = pd.DataFrame(metric_dictionary)
            df.to_csv(os.path.join(path_to_save, 'metrics.csv'), index=False)

            sitk.WriteImage(sitk_pred,  os.path.join(path_to_save, pat_id, (pat_id +'_pr.seg.nrrd')), useCompression=True)

    print("Hello, world!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument("-p", "--path", type=str, required=False, help="path to the config file", default='config.yaml')
    args = parser.parse_args()
    main(args)