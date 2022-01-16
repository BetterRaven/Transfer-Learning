from torch.autograd import Function


class ReverseLayerF(Function):
    # 梯度反转。本来神经网路的目标是向梯度下降的方向优化，相当于最小化目标函数。
    # 这里把梯度反转了，相当于最大化目标函数
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
