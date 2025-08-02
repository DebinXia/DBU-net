import torch
import xlwt
import copy
import os
import re
import numpy as np
import SimpleITK as sitk
import skimage.metrics as metrics
import tifffile
import sklearn.metrics as skm
from tqdm import tqdm


def calc_metrics(fixed_path, moving_path):
    image_1, image_2 = tifffile.imread(fixed_path), tifffile.imread(moving_path)
    MI = mutual_information(image_1, image_2)
    NCC = ncc(image_1.ravel(), image_2.ravel())
    SSIM = ssim(image_1.astype(np.float32), image_2.astype(np.float32))
    MSE = mse(image_1, image_2)
    NCC_1 = ncc_1(torch.tensor(image_1.astype(np.float32)), torch.tensor(image_2.astype(np.float32)))
    print(MI, NCC, SSIM, MSE, NCC_1)
    return MI, NCC, SSIM, MSE, NCC_1.item()


def mutual_information(x, y):

    return skm.mutual_info_score(np.reshape(x, -1), np.reshape(y, -1))


def norm_data(data):
    """
    normalize data to have mean=0 and standard_deviation=1
    """
    mean_data=np.mean(data)
    std_data=np.std(data, ddof=1)
    # return (data-mean_data)/(std_data*np.sqrt(data.size-1))
    return (data-mean_data)/(std_data)


def ncc(data0, data1):
    """
    normalized cross-correlation coefficient between two data sets

    Parameters
    ----------
    data0, data1 :  numpy arrays of same size
    """
    # np.mean(np.multiply((img1-np.mean(img1)),(img2-np.mean(img2))))/(np.std(img1)*np.std(img2))
    return (1.0/(data0.size-1)) * np.sum(norm_data(data0)*norm_data(data1))


from torchir.utils import stablestd

def ncc_1(x1, x2, e=1e-10):
    assert x1.shape == x2.shape, "Inputs are not of equal shape"
    cc = ((x1 - x1.mean()) * (x2 - x2.mean())).mean()
    std = stablestd(x1) * stablestd(x2)
    ncc = cc / (std + e)
    return ncc

def ssim(image1, image2):

    # 5. Compute the Structural Similarity Index (SSIM) between the two
    #    images, ensuring that the difference image is returned
    score = metrics.structural_similarity(image1, image2, data_range=65535)
    # diff = (diff * 255).astype("uint8")

    # 6. You can print only the score if you want
    # print("SSIM: {}".format(score))
    return score


def mse(image1, image2):
    return int(np.mean((image1-image2)**2))


def calc_metric_management(fixed_root, moving_root, save_path):
    fixed_list = os.listdir(fixed_root)
    fixed_list = [name for name in fixed_list if '_1' in name and 'tif' in name]
    fixed_list = sorted(fixed_list)
    moving_list = os.listdir(moving_root)
    moving_new_list = []
    moving_list = [name for name in moving_list if '_2' in name and '.tif' in name]
    """for name in moving_list:
        if 'C1_64_R1' not in name:
            if 'C1' not in name:
                if 'C4_64_R3' not in name:
                    moving_new_list.append(name)

    moving_list = moving_new_list"""
    aligned_pair = []
    for fixed_name in fixed_list:
        moving_ = None
        for moving_name in moving_list:

            if moving_name.split('_')[0] in fixed_name:

                aligned_pair.append([os.path.join(fixed_root, fixed_name),
                                     os.path.join(moving_root, moving_name)])
    book = xlwt.Workbook()
    print(aligned_pair)

    sheet = book.add_sheet(os.path.split(save_path)[1], cell_overwrite_ok=True)
    sheet.write(0, 0, 'name')
    sheet.write(0, 1, 'MI')
    sheet.write(0, 2, 'NCC')
    sheet.write(0, 3, 'SSIM')
    sheet.write(0, 4, 'MSE')

    for i, (fixed, moving) in tqdm(enumerate(aligned_pair)):

        MI, NCC, SSIM, MSE, NCC_1 = calc_metrics(fixed, moving)
        # print(MI, NCC, SSIM)
        sheet.write(i+1, 0, f'{os.path.split(fixed)[1]} to {os.path.split(moving)[1]}')
        sheet.write(i+1, 1, MI)
        sheet.write(i + 1, 2, NCC)
        sheet.write(i + 1, 3, SSIM)
        sheet.write(i + 1, 4, MSE)
        sheet.write(i + 1, 5, NCC_1)
    book.save(save_path + '.xls')


if __name__ == '__main__':
    name_list = ['unet_050', 'unet_100', 'transunet_050', 'transunet_100', 'vitvnet_050', 'vitvnet_100',
                 'mtsunet_100', 'mtsunet_150', 'elastix']
    for name in name_list:
        fixed_root = r'S:\Debin\RawData\Registration\WT2_R123_256_v2'
        moving_path = r'S:\Debin\RawData\Registration\R123_label_v2'.format(name)
        save_root = r'S:\Debin\RawData\Registration\Metrics\raw'.format(name)
        calc_metric_management(fixed_root, moving_path, save_root)