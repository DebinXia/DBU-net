import gc
import csv
from sklearn.metrics import f1_score
import os
import tifffile


def calc_dice(y_true, y_pred):
    # Dice 系数和 F1-score 在二分类中是等价的
    dice = f1_score(y_true.flatten(), y_pred.flatten())
    return dice


def dice_management(fixed_root, moving_root, save_root):
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    fixed_list = sorted([n[:5] for n in os.listdir(fixed_root) if '_1.tif' in n])
    metrics_list = []
    metrics_list.append(['name', 'dice'])
    for slice_id in fixed_list:
        print(slice_id)
        fixed_image = tifffile.imread(os.path.join(fixed_root, slice_id + '_1.tif'))
        moving_image = tifffile.imread(os.path.join(moving_root, slice_id + '_2.tif'))
        metrics_list.append([slice_id, calc_dice(fixed_image, moving_image)])
        del fixed_image, moving_image
        gc.collect()

    with open(os.path.join(save_root + '_dice.csv'), 'w', newline="", ) as file:
        writer = csv.writer(file)
        for row in metrics_list:
            writer.writerow(row)


if __name__ == '__main__':
    name_list = ['unet_050', 'unet_100', 'transunet_050', 'transunet_100', 'vitvnet_050', 'vitvnet_100', 'mtsunet_100',
                 'mtsunet_150']
    name_list = ['elastix', 'raw']
    for name in name_list:
        fixed_root = r'S:\Debin\RawData\Registration\WT2_Seg\seg'
        moving_root = r'S:\Debin\RawData\Registration\WT2_Seg\predict_seg\{}'.format(name)
        save_root = r'S:\Debin\RawData\Registration\WT2_Seg\predict_seg\Metrics\{}'.format(name)
        dice_management(fixed_root, moving_root, save_root)