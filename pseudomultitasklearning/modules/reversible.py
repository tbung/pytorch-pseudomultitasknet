import torch
import torch.nn as nn
import torch.nn.functional as F
from .revnet import RevBlock
from .naive_bayes import GaussianNaiveBayes
from .nearest_mean import NearestMean

def squeeze(x):
    return x.view(-1, 4*x.size(1), x.size(2)//2, x.size(3)//2)

def unsqueeze(x):
    return x.view(-1, x.size(1)//4, x.size(2)*2, x.size(3)*2)

class ReversibleMultiTaskNet(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.name = 'ReversibleMultiTaskNet'

        self.activations = []

        self.revblock2 = RevBlock(4, 4, self.activations, unsqueeze=True)
        self.revblock21 = RevBlock(4, 4, self.activations)
        self.revblock22 = RevBlock(4, 4, self.activations)
        self.revblock3 = RevBlock(16, 16, self.activations, unsqueeze=True)
        self.revblock31 = RevBlock(16, 16, self.activations)
        self.revblock32 = RevBlock(16, 16, self.activations)

        self.nb  = GaussianNaiveBayes(16*7*7,10)
        self.nm  = NearestMean(16*7*7,10)
        self.sm  = nn.LogSoftmax(dim=0)

    def no_class_forward(self, x):
        out = squeeze(x)
        out = self.revblock2(out)
        out = self.revblock21(out)
        out = self.revblock22(out)
        out = squeeze(out)
        out = self.revblock3(out)
        out = self.revblock31(out)
        out = self.revblock32(out)

        return out

    def generate(self, y):
        x = self.revblock32.generate(y)
        x = self.revblock31.generate(x)
        x = self.revblock3.generate(x)
        x = unsqueeze(x)
        x = self.revblock22.generate(x)
        x = self.revblock21.generate(x)
        x = self.revblock2.generate(x)
        x = unsqueeze(x)

        return x

    def free(self):
        """
        Function to clear saved activation residue and thereby free memory
        """
        del self.activations[:]

    def forward(self, x):
        out = self.no_class_forward(x)
        self.activations.append(out.data)
        out = out.view(out.size(0), -1)
        prob_nb = torch.exp(self.nb(out))
        prob_nm = torch.exp(self.sm(self.nm(out)))

        out = out[:,:10]
        
        prob_sm = torch.exp(self.sm(out))
        return prob_nb, prob_sm, prob_nm
