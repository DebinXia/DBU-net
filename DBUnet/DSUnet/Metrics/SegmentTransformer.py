import SimpleITK as sitk
import os
import tifffile
import torch
import numpy as np
import torch.nn.functional as F


def elastix_tf(moving_segment_path, displacement_field_path, save_root):
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    if os.path.isfile(os.path.join(save_root, os.path.split(moving_segment_path)[1])):
        return 0
    moving = tifffile.imread(moving_segment_path)
    moving = sitk.GetImageFromArray(moving)
    flow = tifffile.imread(displacement_field_path)
    flow = sitk.Cast(sitk.GetImageFromArray(flow), sitk.sitkVectorFloat64)
    print(moving.GetSize(), flow.GetSize())
    flow = sitk.DisplacementFieldTransform(flow)
    new_moving = sitk.Resample(moving, moving, flow, sitk.sitkNearestNeighbor)
    sitk.WriteImage(new_moving, os.path.join(save_root, os.path.split(moving_segment_path)[1]))


def torch_tf(moving_segment_path, displacement_field_path, save_root):
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    moving = torch.tensor(tifffile.imread(moving_segment_path).astype(np.float32))[None, None, ]
    flow = torch.tensor(tifffile.imread(displacement_field_path).astype(np.float32))[None, ]
    new_moving = F.grid_sample(moving, flow, mode='nearest',
                  padding_mode='zeros', align_corners=True)

    tifffile.imwrite(os.path.join(save_root, os.path.split(moving_segment_path)[1]),
                    np.clip(np.array(new_moving), 0, 65535).astype(np.uint16))


def segment_transformer_management(moving_root, displacement_root, save_root, type='elastix'):
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    img_id_list = sorted([n[:5] for n in os.listdir(moving_root) if '_2.tif' in n])
    for img_id in img_id_list:
        print(img_id)
        if type == 'elastix':
            elastix_tf(os.path.join(moving_root, '{}_2.tif'.format(img_id)),
                       os.path.join(displacement_root, '{}_4.tif'.format(img_id)),
                       save_root)
        else:
            torch_tf(os.path.join(moving_root, '{}_2.tif'.format(img_id)),
                     os.path.join(displacement_root, '{}_4.tif'.format(img_id)),
                     save_root)


if __name__ == '__main__':
    name_list = ['unet_050', 'unet_100', 'transunet_050', 'transunet_100', 'vitvnet_050', 'vitvnet_100', 'mtsunet_100',
                 'mtsunet_150']
    for name in name_list:
        moving_root = r'S:\Debin\RawData\Registration\WT2_Seg\seg'
        displacement_root = r'S:\Debin\RawData\Registration\WT2_Seg\predict\{}'.format(name)
        save_root = r'S:\Debin\RawData\Registration\WT2_Seg\predict_seg\{}'.format(name)
        segment_transformer_management(moving_root, displacement_root, save_root, type='torch')