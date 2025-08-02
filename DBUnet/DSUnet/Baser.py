import SimpleITK as sitk
import tifffile
import os
import torch
import torch.optim as optim
from torch.utils import data
from Registration.MTSNetwork.loss import NCC, MSE, Smooth
from Registration.MTSNetwork.Model.Morph import MorphModel
from Dataset import Dataset
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
import numpy as np
import gc


default_param = {
    'network': 'voxelmorph',
    'dataset': 'dataset',
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
    'network_param': {'ndim': 2,
                     'in_channels':2,
                     'out_channel': 8,
                     'input_size':[3500, 2500],
                     'device': 'cpu'
    },
    'root': {
        'model_root': 'R:',
        'dataset_root': 'R:',
        'pred_data_root': 'R:',
        'pred_save_root': 'R:',
    },
}


class Baser:
    def __init__(self, param=default_param):
        """
        用作加载文件，实现训练和预测网络
        所有的网络类需要load model 、save model、forward
        所有的损失函数类需要初始化和loss函数
        Args:
            param:
        """
        # self.network_type = default_network[param['network']]
        self.network_param = param['network_param']

        self.device = param['device']
        self.root = param['root']
        self.optimizer_param = param['optimizer_param']
        self.dataset_param = param['dataset＿param']
        self.network = None
        self.optimizer = None
        self.init_network()

    def init_network(self):
        """"""
        # self.network_param['device'] = self.device
        self.network = MorphModel(**self.network_param)
        self.network = self.network.to(self.device)
        print(self.network)
        """self.optimizer = optim.SGD(
            self.network.parameters(), lr=self.optimizer_param['learning_rate'], momentum=0.99)"""
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=self.optimizer_param['learning_rate'])

    def save_network(self, save_root, epoch):
        save_file_name = os.path.join(save_root, 'last_{:03d}.pth'.format(epoch))
        torch.save(self.network.state_dict(), save_file_name)

    def load_network(self, save_root, epoch=99):
        model_path = os.path.join(save_root, 'last_{:03d}.pth'.format(epoch))
        self.network.load_state_dict(torch.load(model_path))

    def logging(self, save_root, tran_loss_list):
        import csv
        with open(os.path.join(save_root, 'training_log.csv'), mode='a', newline='') as file:
            writer = csv.writer(file)
            # writer.writerow(['Epoch', 'Train_Loss', 'Train_Metric', 'Test_Loss', 'Test_Metric'])
            for train_loss in tran_loss_list:
                writer.writerow(train_loss)
        return []

    def adjust_learning_rate(self, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)

    def load_dataset(self,):
        dataset_params = self.dataset_param
        # Generators
        image_list = os.listdir(self.root['dataset_root'])
        image_list = [m for m in image_list if 'tif' in m]
        filename = list(set([x.split('_')[0]
                             for x in image_list]))
        # filename = filename[:100]
        # filename = ['00011', '00011']
        print(len(filename))
        partition = {}
        partition['train'], partition['validation'] = train_test_split(
            filename, test_size=0.20, random_state=42)
        # partition['train'] = filename
        training_set = Dataset(partition['train'], self.root['dataset_root'])
        training_generator = data.DataLoader(training_set, **dataset_params)

        validation_set = Dataset(partition['validation'], self.root['dataset_root'])
        validation_generator = data.DataLoader(validation_set, **dataset_params)

        return training_set, validation_set, training_generator, validation_generator

    def train_model(self):
        """训练模型"""
        training_set, validation_set, training_generator, validation_generator = self.load_dataset()
        train_loss_list = []
        for epoch in range(self.optimizer_param['max_epochs']):
            start_time = time.time()
            train_loss = 0
            train_dice_score = 0
            val_loss = 0
            val_dice_score = 0

            for batch_fixed, batch_moving in tqdm(training_generator):
                loss, dice = self.grad_step(batch_fixed, batch_moving, train_model=True)
                # print(dice)
                train_dice_score += dice.data
                train_loss += loss.data
            print('[', "{0:.2f}".format((time.time() - start_time) / 60), 'mins]', 'After', epoch + 1,
                  'epochs, the Average training loss is ', train_loss *
                  self.dataset_param['batch_size'] / len(training_set), 'and average NCC score is',
                  train_dice_score.data * self.dataset_param['batch_size'] / len(training_set))
            for batch_fixed, batch_moving in tqdm(validation_generator):
                # Transfer to GPU
                # loss, dice = self.train_step(batch_fixed, batch_moving)
                loss, dice = self.grad_step(batch_fixed, batch_moving, train_model=False)
                val_dice_score += dice.data
                val_loss += loss.data
            print('[', "{0:.2f}".format((time.time() - start_time) / 60), 'mins]', 'After', epoch + 1,
                  'epochs, the Average validations loss is ', val_loss *
                  self.dataset_param['batch_size'] / len(validation_set), 'and average NCC score is',
                  val_dice_score.data * self.dataset_param['batch_size'] / len(validation_set))

            train_loss_list.append(['epoch {:04d}'.format(epoch), train_loss *
                  self.dataset_param['batch_size'] / len(training_set),
                                   train_dice_score.data * self.dataset_param['batch_size'] / len(training_set),
                                    val_loss *
                                    self.dataset_param['batch_size'] / len(validation_set),
                                    val_dice_score.data * self.dataset_param['batch_size'] / len(validation_set)])
            self.save_network(self.root['model_root'], epoch)
            train_loss_list = self.logging(self.root['model_root'], train_loss_list)

    def grad_step(self, batch_fixed, batch_moving, return_metric_score=True, train_model=True):
        """这个函数是训练和预测的主要函数，用作形变数据，获取loss， 反向传播
        """
        if train_model:
            self.optimizer.zero_grad()
            batch_fixed, batch_moving = batch_fixed.to(
                self.device), batch_moving.to(self.device)
            registered_image, field = self.network(batch_fixed, batch_moving)

            a = NCC()
            b = Smooth()
            # c = MSE()

            train_loss = a.loss(batch_fixed, registered_image[-1]) + b.loss(field[-1])  # c.loss(field, label)
            if len(field) >= 2:
                for i in range(0, len(field)-1):
                    train_loss += 0.5 ** (i+1) * a.loss(batch_fixed, registered_image[i])
            train_loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
            if return_metric_score:
                train_dice_score = NCC.metric(batch_fixed, registered_image[-1])
                return train_loss, train_dice_score
            return registered_image.detach(), train_loss.detach(), field.detach()
        else:
            with torch.set_grad_enabled(False):
                batch_fixed, batch_moving = batch_fixed.to(
                    self.device), batch_moving.to(self.device)
                registered_image, field = self.network(batch_fixed, batch_moving, )

                val_dice_score = NCC.metric(registered_image[-1], batch_fixed)
                return 1 - val_dice_score, val_dice_score

    def pred_step(self, batch_fixed, batch_moving):
        batch_fixed, batch_moving = batch_fixed.to(
            self.device), batch_moving.to(self.device)
        registered_image, para = self.network(batch_fixed, batch_moving)
        a = NCC()
        train_loss = a.metric(registered_image[-1], batch_fixed,)
        # print(a.loss(batch_fixed, batch_moving), train_loss)
        return registered_image[-1].cpu().detach(), train_loss.cpu(), para[-1].cpu()

    def pred(self, epoch=199):
        """对数据集实现预测"""
        if not os.path.exists(self.root['pred_save_root']):
            os.mkdir(self.root['pred_save_root'])
        fixed_list = os.listdir(self.root['pred_data_root'])
        fixed_list = [fixed_name for fixed_name in fixed_list if '_1.tif' in fixed_name]
        # fixed_list = [fixed_name for fixed_name in fixed_list if '1000' in fixed_name]
        fixed_list = sorted(fixed_list)
        # moving_list = os.listdir(moving_root)
        # moving_list = [moving_name for moving_name in moving_list if '405' in moving_name]
        a = []

        self.load_network(self.root['model_root'], epoch=epoch)
        with torch.set_grad_enabled(False):
            for fixed_name in tqdm(fixed_list):
                # print(fixed_name)
                fixed_image = torch.Tensor(
                    sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root['pred_data_root'],
                                                                       fixed_name))).astype(float)).unsqueeze(
                    dim=0).unsqueeze(dim=0)

                moving_image = \
                    torch.Tensor(sitk.GetArrayFromImage(
                        sitk.ReadImage(os.path.join(self.root['pred_data_root'], fixed_name[:-5] + '2.tif'))).astype(
                        float)).unsqueeze(
                        dim=0).unsqueeze(dim=0)
                """tifffile.imwrite(os.path.join(self.root['pred_save_root'], fixed_name[:-5] + '1_0.tif'),
                                 fixed_image[0, 0].numpy())
                tifffile.imwrite(os.path.join(self.root['pred_save_root'], fixed_name[:-5] + '2_0.tif'),
                                 moving_image[0, 0].numpy())"""

                new_moving_image, loss, param = self.pred_step(fixed_image, moving_image)
                # print(fixed_name, loss)
                tifffile.imwrite(os.path.join(self.root['pred_save_root'], fixed_name[:-5] + '3.tif'),
                                 np.clip(new_moving_image.float().squeeze().squeeze().numpy(), 0, 65535).astype(np.uint16))
                tifffile.imwrite(os.path.join(self.root['pred_save_root'], fixed_name[:-5] + '4.tif'),
                                 param[:, ].float().squeeze().squeeze().numpy())
                del fixed_image, moving_image, new_moving_image
                gc.collect()
                a.append(loss)
        print(np.mean(np.array(a)))