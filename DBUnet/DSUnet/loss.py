import torch


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Smooth(torch.nn.Module):
    def __init__(self):
        super(Smooth).__init__()

    def forward(self, y_field):
        """
        平滑形变场
        Args:
            y_field:

        Returns:

        """
        if len(y_field.shape) == 4:
            dy = y_field[:, :, 1:, :] - y_field[:, :, :-1, :]
            dx = y_field[:, :, :, 1:] - y_field[:, :, :, :-1]

            dx = torch.mul(dx, dx)
            dy = torch.mul(dy, dy)
            d = torch.mean(dx) + torch.mean(dy)
            return d / 2.0
        else:
            dy = y_field[:, :, :, 1:, :] - y_field[:, :, :, :-1, :]
            dx = y_field[:, :, :, :, 1:] - y_field[:, :, :, :, :-1]
            dz = y_field[:, :, 1:, :, :] - y_field[:, :, :-1, :, :]
            dx = torch.mul(dx, dx)
            dy = torch.mul(dy, dy)
            dz = torch.mul(dz, dz)
            d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
            return d / 3.0

    def loss(self, y_field):
        return self.forward(y_field)


class StableStd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        assert tensor.numel() > 1
        ctx.tensor = tensor.detach()
        res = torch.std(tensor).detach()
        ctx.result = res.detach()
        return res

    @staticmethod
    def backward(ctx, grad_output):
        tensor = ctx.tensor.detach()
        result = ctx.result.detach()
        e = 1e-6
        assert tensor.numel() > 1
        return (
            (2.0 / (tensor.numel() - 1.0))
            * (grad_output.detach() / (result.detach() * 2 + e))
            * (tensor.detach() - tensor.mean().detach())
        )


stablestd = StableStd.apply

class NCC(torch.nn.Module):
    # 结果还没有与标准结果验证，但是作为配准指标效果不错，可以收敛
    def __init__(self, use_mask: bool = False):

        super().__init__()
        if use_mask:
            self.mask = self.masked_metric
        else:
            self.mask = self.metric

    @staticmethod
    def ncc(x1, x2, e=1e-10):
        assert x1.shape == x2.shape, "Inputs are not of equal shape"
        cc = ((x1 - x1.mean()) * (x2 - x2.mean())).mean()
        std = stablestd(x1) * stablestd(x2)
        ncc = cc / (std + e)
        return ncc

    @staticmethod
    def ncc_mask(x1, x2, mask, e=1e-10):  # TODO: calculate ncc per sample
        assert x1.shape == x2.shape, "Inputs are not of equal shape"
        x1 = torch.masked_select(x1, mask)
        x2 = torch.masked_select(x2, mask)
        cc = ((x1 - x1.mean()) * (x2 - x2.mean())).mean()
        std = stablestd(x1) * stablestd(x2)
        ncc = cc / (std + e)
        return ncc

    def loss(self, fixed: torch.Tensor, warped: torch.Tensor) -> torch.Tensor:
        return 1 - self.ncc(fixed, warped)

    def masked_loss(self, fixed: torch.Tensor, warped: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return 1 - self.ncc_mask(fixed, warped, mask)
    @staticmethod
    def metric(fixed: torch.Tensor, warped: torch.Tensor) -> torch.Tensor:
        return NCC.ncc(fixed, warped)

    def forward(self, fixed: torch.Tensor, warped: torch.Tensor, grid=None):
        if self.mask is True:
            return self.masked_loss(fixed, warped)
        else:
            return self.loss(fixed, warped)