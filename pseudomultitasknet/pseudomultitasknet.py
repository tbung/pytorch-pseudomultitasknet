import torch
import torch.nn as nn
import torch.nn.functional as F

from binet import BidirectionalBlock
from pseudomultitasknet.modules import GaussianNaiveBayes, NearestMean


def squeeze(x, factor=2):
    batch_size, channels, in_height, in_width = x.size()
    channels *= factor**2

    out_height = in_height // factor
    out_width = in_width // factor

    input_view = x.contiguous().view(
        batch_size, -1, out_height, factor, out_width, factor
    )

    out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()

    return out.view(batch_size, channels, out_height, out_width)


def unsqueeze(x, factor=2):
    return F.pixel_shuffle(x, factor)


class PseudoMultiTaskNet(nn.Module):
    def __init__(self, no_svd=False):
        super(self.__class__, self).__init__()
        self.name = 'ReversibleMultiTaskNet'

        self.no_svd = no_svd

        self.activations = []

        self.padding = nn.ZeroPad2d(2)

        self.revblock2 = BidirectionalBlock(4, 4, self.activations,
                                            storage_hooks=[unsqueeze],
                                            no_activation=True)
        self.revblock21 = BidirectionalBlock(4, 4, self.activations)
        self.revblock22 = BidirectionalBlock(4, 4, self.activations,
                                             inverse_hooks=[squeeze])
        self.revblock3 = BidirectionalBlock(16, 16, self.activations,
                                            storage_hooks=[unsqueeze],
                                            no_activation=True)
        self.revblock31 = BidirectionalBlock(16, 16, self.activations)
        self.revblock32 = BidirectionalBlock(16, 16, self.activations,
                                             inverse_hooks=[squeeze])
        self.revblock4 = BidirectionalBlock(64, 64, self.activations,
                                            storage_hooks=[unsqueeze],
                                            no_activation=True)
        self.revblock41 = BidirectionalBlock(64, 64, self.activations)
        self.revblock42 = BidirectionalBlock(64, 64, self.activations,
                                             inverse_hooks=[squeeze])
        self.revblock5 = BidirectionalBlock(256, 256, self.activations,
                                            storage_hooks=[unsqueeze],
                                            no_activation=True)
        self.revblock51 = BidirectionalBlock(256, 256, self.activations)
        self.revblock52 = BidirectionalBlock(256, 256, self.activations)

        self.size = 32*32

        self.nb = GaussianNaiveBayes(self.size, 10)
        self.nm = NearestMean(self.size, 10)
        self.sm = nn.LogSoftmax(dim=0)

    def inversible_forward(self, x):
        out = self.padding(x)
        out = squeeze(out)
        out = self.revblock2(out)
        out = self.revblock21(out)
        out = self.revblock22(out)
        out = squeeze(out)
        out = self.revblock3(out)
        out = self.revblock31(out)
        out = self.revblock32(out)
        out = squeeze(out)
        out = self.revblock4(out)
        out = self.revblock41(out)
        out = self.revblock42(out)
        out = squeeze(out)
        out = self.revblock5(out)
        out = self.revblock51(out)
        out = self.revblock52(out)

        return out

    def orthogonal(self, x):
        U, S, V = torch.svd(x, some=True)
        out = torch.mm(x, V)
        out = F.pad(out, (0, self.size - out.size(1), 0, 0))
        return out, S, V

    def generate(self, y):
        x = self.revblock52.inverse_forward(y)
        x = self.revblock51.inverse_forward(x)
        x = self.revblock5.inverse_forward(x)
        x = unsqueeze(x)
        x = self.revblock42.inverse_forward(x)
        x = self.revblock41.inverse_forward(x)
        x = self.revblock4.inverse_forward(x)
        x = unsqueeze(x)
        x = self.revblock32.inverse_forward(x)
        x = self.revblock31.inverse_forward(x)
        x = self.revblock3.inverse_forward(x)
        x = unsqueeze(x)
        x = self.revblock22.inverse_forward(x)
        x = self.revblock21.inverse_forward(x)
        x = self.revblock2.inverse_forward(x)

        self.activations.append(x.data)

        x = unsqueeze(x)
        return x

    def orthogonal_generate(self, y_, V):
        y = torch.mm(y_, V)
        y = y.view(-1, 16, 7, 7)
        return self.generate(y)

    def free(self):
        """Clear saved activation residue and thereby free memory."""
        del self.activations[:]

    def forward(self, x):
        out = self.inversible_forward(x)
        self.activations.append(out.data)
        out = out.view(out.size(0), -1)
        if not self.no_svd:
            _, S, V = self.orthogonal(out)
        else:
            S = None
        prob_nb = torch.exp(self.nb(out))
        prob_nm = torch.exp(self.sm(self.nm(out)))

        out = out[:, :10]

        prob_sm = torch.exp(self.sm(out))
        return prob_nb, prob_sm, prob_nm, S
