import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    这个空间变换器的主要作用是将深度学习（unet）产生的形变场指导moving图像进行变换，并且将梯度反向传播。
    """

    def __init__(self, size=[3500, 2500], mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        # 创建原来的坐标对应网格。
        self.size = size
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    @staticmethod
    def identity_grid(shape, stackdim, dtype=torch.float32, device="cpu"):
        """Create and identity grid.
        初始化形变场"""
        tensors = (torch.arange(s, dtype=dtype, device=device) for s in shape)
        return torch.stack(
            torch.meshgrid(*tensors)[::-1], stackdim
        )  # z,y,x shape and flip for x, y, z coords

    def __grid__(self, flow):
        new_grid = flow + self.identity_grid(
            flow.shape[2:], stackdim=0, dtype=flow.dtype, device=flow.device
        )

        max_extent = (
                torch.tensor(
                    new_grid.shape[2:][::-1], dtype=new_grid.dtype, device=new_grid.device
                )
                - 1
        )
        # print(max_extent)
        if len(flow.shape[2:]) == 3:
            new_grid = new_grid.permute(0, 2, 3, 4, 1)
        else:
            new_grid = new_grid.permute(0, 2, 3, 1)

        # return 2 * (new_grid / max_extent) - 1
        return new_grid

    def forward(self, src, flow):
        # flow表示需要进行的偏移量，而grid表示原来需要对应的偏移量，意味着flow可以直接用SimpleITK进行形变

        new_grid = flow + self.identity_grid(
            flow.shape[2:], stackdim=0, dtype=flow.dtype, device=flow.device
        )
        max_extent = (
                torch.tensor(
                    new_grid.shape[2:][::-1], dtype=new_grid.dtype, device=new_grid.device
                )
                - 1
        )
        # print(max_extent)
        if len(flow.shape[2:]) == 3:
            new_grid = new_grid.permute(0, 2, 3, 4, 1)
        else:
            new_grid = new_grid.permute(0, 2, 3, 1)

        return F.grid_sample(src, 2 * (new_grid / max_extent) - 1, mode=self.mode,
                             padding_mode='zeros', align_corners=True), 2 * (new_grid / max_extent) - 1

    def transformer(self, src, grid):
        new_grid = grid.permute(0, 2, 3, 4, 1)
        return F.grid_sample(src, new_grid, mode=self.mode,
                             padding_mode='zeros', align_corners=True)