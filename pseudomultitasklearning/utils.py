import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def imshow(img):
    """Plot a 4D tensor as a batch of images
    
    Arguments:
        img (Tensor): The images to display
    """

    plt.figure(tight_layout=True)

    img = (img - img.min())/(img.max()-img.min())
    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
