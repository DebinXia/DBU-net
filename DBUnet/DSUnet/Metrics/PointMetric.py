import os
import numpy as np
import tifffile
from skimage import measure
import csv
from tqdm import tqdm

def get_properties(image):
    label = measure.label(image)
    properties = measure.regionprops(label)
    centroids_list = []
    for prop_i in tqdm(properties):
        if prop_i.area < 8:
            continue

        centroids_list.append(prop_i.centroid)

    return np.array(centroids_list)


def min_dist(fixed_centroid_array, moving_centroid_array):
    fixed_distance_array = np.zeros((fixed_centroid_array.shape[0]), dtype=np.float32)
    for p_i in range(fixed_centroid_array.shape[0]):
        dist = np.sqrt(np.sum(np.square(moving_centroid_array - fixed_centroid_array[p_i]), axis=1))
        if np.min(dist)>20:
            fixed_distance_array[p_i] = 100000
        else:
        # a = np.where(dist == np.min(dist))
            fixed_distance_array[p_i] = np.where(dist == np.min(dist))[0]

    return fixed_distance_array


def match_point_1(fixed_distance_array, moving_distance_array):
    match_points_list = []
    for i in range(fixed_distance_array.shape[0]):
        if int(fixed_distance_array[i]) > moving_distance_array.shape[0]:
            continue
        if moving_distance_array[int(fixed_distance_array[i])] == i:
            match_points_list.append([i, fixed_distance_array[i]])
    return match_points_list


def centroids_distance(fixed_path, moving_path, ):
    fixed_image = tifffile.imread(fixed_path)
    moving_image = tifffile.imread(moving_path)# [0, 0]

    fixed_centroid_array = get_properties(fixed_image)
    moving_centroid_array = get_properties(moving_image)
    if len(fixed_centroid_array) <= 0 or len(moving_centroid_array) == 0:
        return None
    fixed_distance_array = min_dist(fixed_centroid_array, moving_centroid_array)
    moving_distance_array = min_dist(moving_centroid_array, fixed_centroid_array)

    match_points_list = match_point_1(fixed_distance_array, moving_distance_array)
    match_points_pair = np.zeros(len(match_points_list))
    for p_i, pairs in enumerate(match_points_list):
        match_points_pair[p_i] = np.sqrt(np.sum(np.square(fixed_centroid_array[int(pairs[0])] -
                                  moving_centroid_array[int(pairs[1])])))
    return match_points_pair


def centroid_match_management(fixed_root, moving_root, save_root):
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    fixed_list = sorted([n for n in os.listdir(fixed_root) if '.tif' in n and '_1' in n])
    moving_list = sorted([n for n in os.listdir(moving_root) if '.tif' in n and '_2' in n])
    match_info_list = []
    whole_info_list = []
    for moving_name in moving_list:

        fixed_image_name = None
        for name in fixed_list:
            if name[:6] in moving_name:
                fixed_image_name = name
        if fixed_image_name is None:
            continue
        print(fixed_image_name, moving_name)
        try:
            match_points_pair = centroids_distance(os.path.join(fixed_root, fixed_image_name),
                                                   os.path.join(moving_root, moving_name), )
        except:
            continue
        if match_points_pair is None:
            continue
        for iii in range(match_points_pair.shape[0]):
            whole_info_list.append(match_points_pair[iii].astype(np.float32))
        match_info_list.append([fixed_image_name[:6], match_points_pair.shape[0], np.mean(match_points_pair)])

    with open(os.path.join(save_root, 'R1_R4_match_points_distance.csv'), 'w', newline="") as file:
        writer = csv.writer(file)
        writer.writerow(['name', 'match points number', 'mean distance'])
        for info in match_info_list:
            writer.writerow(info)

    """with open(save_root + '_points_distance.csv', 'w', newline="") as file:
        writer = csv.writer(file)
        writer.writerow(['distance'])
        for info in whole_info_list:
            writer.writerow([info])"""


if __name__ == '__main__':
    name_list = ['unet_050', 'unet_100', 'transunet_050', 'transunet_100', 'vitvnet_050', 'vitvnet_100', 'mtsunet_100',
                 'mtsunet_150']
    name_list = ['elastix']
    for name in name_list:
        fixed_root = r'S:\Debin\RawData\Registration\WT2_Seg\seg'
        moving_path = r'S:\Debin\RawData\Registration\WT2_Seg\predict_seg\{}'.format(name)
        save_root = r'S:\Debin\RawData\Registration\WT2_Seg\predict_seg\Metrics\{}'.format(name)
        centroid_match_management(fixed_root, moving_path, save_root)