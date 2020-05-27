import numpy as np
from scipy.interpolate import RectBivariateSpline

from InverseCompositionAffine import InverseCompositionAffine
from LucasKanadeAffine import LucasKanadeAffine
from scipy.ndimage import affine_transform
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import matplotlib.pyplot as plt

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)
    # M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    M = InverseCompositionAffine(image1, image2, threshold, num_iters)

    M = np.append(M, np.asarray([[0, 0, 1]]), axis=0)
    image2_warped = affine_transform(image2, M, image1.shape)
    difference = abs(image1 - image2_warped)
    mask[(difference <= tolerance)] = 0
    mask[(image2_warped == 0)] = 0

    # ants
    mask = binary_dilation(mask, structure=np.ones((4, 4)), iterations=1)
    mask = binary_erosion(mask, structure=np.ones((1, 1)), iterations=1)
    mask[:12, :] = mask[:, :12] = mask[mask.shape[0]-12:, :] = mask[:, mask.shape[1]-12:] = 0

    # aerial
    # mask = binary_dilation(mask, structure=np.ones((5, 5)), iterations=1)
    # mask = binary_erosion(mask, structure=np.ones((3, 3)), iterations=1)
    # mask[:12, :] = mask[:, :12] = mask[mask.shape[0]-12:, :] = mask[:, mask.shape[1]-12:] = 0


    return mask
