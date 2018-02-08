import torch
import numpy as np

from pseudomultitasknet.interpolation_training import InterpolationTraining

class SphericalInterpolation(InterpolationTraining):
    def __init__(self):
        super(SphericalInterpolation, self).__init__()
        self.name = "spherical_interpolation"
        self.aug_lr = 1e-10

    def interpolation(self, outputs):
        o0 = outputs[0].data.view(1, -1)
        o1 = outputs[1].data.view(-1, 1)
        theta = torch.acos(torch.mm(o0, o1)
                             / (o0.norm(1)*o1.norm(1))).cpu().numpy()[0][0]

        sin_t = np.sin(theta)

        steps = np.linspace(0, 1, 16)
        return torch.stack([np.sin((1-t)*theta)/sin_t*outputs.data[0]+np.sin(t*theta)/sin_t*outputs.data[1] for t in steps], dim=0)


if __name__ == "__main__":
    experiment = SphericalInterpolation()
    experiment.run()
