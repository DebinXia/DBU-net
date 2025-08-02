import gc
import os
import SimpleITK as sitk
import tifffile
import numpy as np
from tqdm import tqdm


def main(image_root, save_root):
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    image_list = sorted([n for n in os.listdir(image_root) if 'tif' in n])
    image_list = sorted([n for n in image_list if '_1' in n or '_2' in n])
    for image_name in tqdm(image_list):
        image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(image_root, image_name)))
        image = np.clip(image, 112, 624) - 112
        tifffile.imwrite(os.path.join(save_root, image_name), image)
        del image
        gc.collect()


if __name__ == '__main__':
    image_root = r'R:\DebinXia\RawData\Registration\MultiRoundDataset\WT2\WT2_R123_256_v2'
    save_root = r'R:\DebinXia\RawData\Registration\MultiRoundDataset\WT2\WT2_R123_256_v2_nor'
    main(image_root, save_root)