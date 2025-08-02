from Registration.MTSNetwork.Baser import Baser
import os
import numpy as np


default_param = {
    'device': 'cuda:0',
    'optimizer_param':{
        'max_epochs': 500,
        'learning_rate':0.01
    },

    'dataset＿param': {'batch_size': 2,
                  'shuffle': True,
                  'num_workers': 6,
                  'worker_init_fn': np.random.seed(42)
                  },
    'network_param': {'ndim':3,
                      'in_channel': 2,
                        'out_channel': 3,
                        'input_size': (64, 256, 256), 'network_name':'unet',},
    'root': {
        'model_root': 'R:',
        'dataset_root': 'R:',
        'pred_data_root': 'R:',
        'pred_save_root': 'R:',
    },
}


def traning():
    """训练参数设置"""
    parameter_train = default_param
    # set root
    name = 'vitvnet'
    number = 100
    parameter_train['root']['model_root'] = r'S:\Debin\RawData\Registration\model_v2\{}'.format(name)
    parameter_train['root']['dataset_root'] = r'R:\DebinXia\RawData\Registration\MultiRoundDataset\WT1\label_v2\R123_label_v2'

    parameter_train['root']['pred_data_root'] = r'S:\Debin\RawData\Registration\WT2_Seg\raw'
    parameter_train['root'][
        'pred_save_root'] = r'S:\Debin\RawData\Registration\WT2_Seg\predict\{}_{:03d}'.format(name, number)
    # set network param
    if not os.path.exists(parameter_train['root']['model_root']):
        os.mkdir(parameter_train['root']['model_root'])
    parameter_train['network_param'] = {'ndim': 3,
                                        'in_channel': 2,
                                        'out_channel': 3,
                                        'input_size': (64, 256, 256), 'network_name': '{}'.format(name), 'use_adding':True}
    print(parameter_train)
    if not os.path.exists(parameter_train['root']['pred_save_root']):
        os.mkdir(parameter_train['root']['pred_save_root'])
    tra = Baser(parameter_train)

    # tra.train_model()
    tra.pred(number)


if __name__ == '__main__':

    traning()