import torch.nn as nn
from DBU-net.Model.UNet.DBU_net import ENet as unetmts
import torch
import os
from DBU-net.Model.UNet.STN import SpatialTransformer


class MorphModel(nn.Module):
    def __init__(self, ndim, in_channel=2, out_channel=3, input_size=(64, 256, 256), network_name='unet', use_adding=False):
        super(MorphModel, self).__init__()
        self.ndim = ndim
        self.use_adding = use_adding
        self.unet = self.load_network(network_name, ndim, in_channel, out_channel)
        self.transform = SpatialTransformer(input_size)

    def load_network(self, network_name, ndim, in_channel, out_channel):
        if network_name == 'unet':
            return unet(ndim, in_channel, out_channel)
        elif network_name == 'mtsunet':
            return unetmts(ndim, in_channel, out_channel)

    def save_model(self, save_root, epoch=0):
        save_file_name = os.path.join(save_root, 'last_{:03d}.pth'.format(epoch))
        torch.save(self.unet.state_dict(), save_file_name)

    def load_model(self, model_root, epoch=0):
        model_path = os.path.join(model_root, 'last_{:03d}.pth'.format(epoch))
        self.unet.load_state_dict(torch.load(model_path))

    def forward(self, fixed_image, moving_image):
        x = torch.cat((fixed_image, moving_image), dim=1)
        deformation_matrix = self.unet(x)

        registered_image, field = [], []
        if self.use_adding:
            grid = deformation_matrix[0]
            if len(deformation_matrix) > 1:
                for i in range(1, len(deformation_matrix)):
                    grid += deformation_matrix[i]
            a, b = self.transform(moving_image, grid)
            registered_image.append(a)
            field.append(b)
        else:
            for i in range(len(deformation_matrix)):
                a, b = self.transform(moving_image, deformation_matrix[i])
                registered_image.append(a)
                field.append(b)

        return registered_image, field


if __name__ == '__main__':
    import numpy as np

    fixed = torch.tensor(np.zeros((1, 1, 64, 256, 256), dtype=np.float32)).to('cuda:1')
    moving = torch.tensor(np.zeros((1, 1, 64, 256, 256), dtype=np.float32)).to('cuda:1')
    model = MorphModel(ndim=3, in_channel=2, out_channel=3, input_size=(64, 256, 256), network_name='xmorpher', use_adding=False)
    model = model.to('cuda:1')

    a = model(fixed, moving)
    print(a.Size())
