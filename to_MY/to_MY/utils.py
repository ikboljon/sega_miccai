import os
import json
import torch
import numpy as np
from monai.transforms import MapTransform

class ClipCT(MapTransform):
    """
    Convert labels to multi channels based on hecktor classes:
    label 1 is the tumor
    label 2 is the lymph node

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key == "ct":
                d[key] = torch.clip(d[key], min=-200, max=200)
            # elif key == "pt":
            #     d[key] = torch.clip(d[key], d[key].min(), 5)
        return d

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)

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


def load_model_weights(model, path_to_model_weights):
    model_state_dict = torch.load(path_to_model_weights, map_location=lambda storage, loc: storage)
    state_dict = model_state_dict['state_dict']
    del state_dict['dice_loss.dice.class_weight']
    
    for key in list(state_dict):
        # if dice_loss exists in then state_dict, remove it
        # if key != 'dice_loss.dice.class_weight':
        

        if 'model.' in key:
            state_dict[key.replace("model.", "")] = state_dict.pop(key)

    model.load_state_dict(state_dict)
    return model