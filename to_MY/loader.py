from monai import transforms
from monai import data
from utils import datafold_read


def get_loader(batch_size, sw_batch_size, data_dir, json_list, fold, roi):
    data_dir = data_dir
    datalist_json = json_list

    train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, fold=fold)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image","image2", "label"], ensure_channel_first = True),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(roi[0], roi[1], roi[2]), method='end'),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi,
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            transforms.NormalizeIntensityd(keys=["image", "image2"]),
 
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image","image2", "label"], ensure_channel_first = True),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(roi[0], roi[1], roi[2]), method='end'),
        ]            
    )
    train_ds = data.Dataset(data=train_files, transform=train_transform)


    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_ds = data.Dataset(data=validation_files, transform=val_transform)

    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    return train_loader, val_loader, train_ds, val_ds