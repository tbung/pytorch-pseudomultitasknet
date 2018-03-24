import torch
import torch.nn as nn


class SparsityFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, Q, tau):
        Y = Q.matmul(x.t()).t()
        K = Y.size(0)

        ctx.tau = tau

        loss = 1/K * torch.norm(Y, 1)
        ctx.save_for_backward(x, Y, Q)
        return loss

    @staticmethod
    def backward(ctx, dy):
        x, Y, Q = ctx.saved_variables
        t = ctx.tau

        K = Y.size(0)
        one = torch.eye(2*K, 2*K).cuda()

        U = torch.cat((Y.sign()/(2*K), -Y)).t()
        V_t = torch.cat((Y, Y.sign()/(2*K)))

        dQ_ = 2 * t * U.matmul((one + t * V_t.matmul(U)).inverse()).matmul(V_t)
        dQ = dQ_.matmul(Q)

        with torch.enable_grad():
            x_ = torch.autograd.Variable(x.data, requires_grad=True)
            y_ = x_.matmul(Q)
            loss = 1/K * torch.norm(y_, 1)
            dx = torch.autograd.grad(loss, x_, dy)[0]

        return dx, dQ, None


class SparsityBlock(nn.Module):
    def __init__(self, size, tau):
        super(SparsityBlock, self).__init__()
        self.tau = tau

        self.register_parameter(
            'Q',
            nn.Parameter(torch.Tensor(size, size))
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal(self.Q)

    def forward(self, x):
        return SparsityFunction.apply(x, self.Q, self.tau)
