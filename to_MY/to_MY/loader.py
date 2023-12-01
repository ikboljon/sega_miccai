from monai import transforms
from monai import data
from utils import datafold_read, ClipCT


def get_loader(batch_size=1, data_dir='/', json_list='/json.json', fold=0, roi=(96,96,96)):
    data_dir = data_dir
    datalist_json = json_list

    train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, fold=fold)
    # train_transform = transforms.Compose(
    #     [
    #         transforms.LoadImaged(keys=["ct", "pt", "seg"], ensure_channel_first = True),
    #         transforms.ConcatItemsd(['pt', 'ct'], 'ctpt'),
    #         transforms.SpatialPadd(keys=["ctpt", "seg"], spatial_size=(roi[0], roi[1], roi[2]), method='end'),
    #         transforms.RandCropByPosNegLabeld(
    #             keys=["ctpt", "seg"],
    #             label_key="seg",
    #             spatial_size=roi,
    #             pos=1,
    #             neg=1,
    #             num_samples=1,
    #             image_key="ctpt",
    #             image_threshold=0,
    #         ),
    #         transforms.NormalizeIntensityd(keys=["ctpt", "seg"]),
 
    #     ]
    # )

    val_transform = transforms.Compose(
        [
            # transforms.LoadImaged(keys=["ct", "pt", "seg"], ensure_channel_first = True, image_only=True),
            # transforms.ConcatItemsd(['pt', 'ct'], 'ctpt'),
            # transforms.SpatialPadd(keys=["ctpt", "seg"], spatial_size=(roi[0], roi[1], roi[2]), method='end'),
        
            transforms.LoadImaged(keys=["ct", "pt", "seg"], ensure_channel_first = True),
            transforms.SpatialPadd(keys=["ct", "pt", "seg"], spatial_size=(200,200,310), method='end'),
            transforms.Orientationd(keys=["ct", "pt", "seg"], axcodes="PLS"),
            transforms.NormalizeIntensityd(keys=["pt"]),
            ClipCT(keys=["ct"]),
            transforms.ScaleIntensityd(keys=["ct"], minv=0, maxv=1),
            #MulPTFM(keys=["ct","pt"]),
            transforms.ConcatItemsd(keys=["pt", "ct"], name="ctpt"),
        ]            
    )
    # train_ds = data.Dataset(data=train_files, transform=train_transform)


    # train_loader = data.DataLoader(
    #     train_ds,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=8,
    #     pin_memory=True,
    # )
    val_ds = data.Dataset(data=validation_files, transform=val_transform)

    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # return train_loader, val_loader, train_ds, val_ds
    return val_loader, val_ds