import torch
import torch.nn as nn


class SparsityFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, Q, tau):
        Y = Q.matmul(x.t()).t()

        ctx.save_for_backward(Y, Q)
        ctx.tau = tau

        return torch.norm(Y, 1)

    @staticmethod
    def backward(ctx, dy):
        Y, Q = ctx.saved_variables
        t = ctx.tau

        K = Y.size(0)
        one = torch.eye(2*K, 2*K)

        U = torch.cat((Y.sign()/(2*K), -Y))
        V_t = torch.cat((Y, Y.sign()/(2*K))).t()

        dQ = 2 * t * U.matmul((one + t * V_t.matmul(U)).inverse()).matmul(V_t).matmul(Q)

        dx = Q.matmul(dy.t()).t()

        return dx, dQ, None


class SparsityBlock(nn.Module):
    def __init__(self, size, tau):
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
