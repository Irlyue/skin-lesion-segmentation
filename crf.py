import numpy as np

from pydensecrf.densecrf import DenseCRF2D
from pydensecrf.utils import unary_from_softmax


def crf_post_process(image, prob, n_steps=5):
    """
    Use CRF as a post processing technique.

    :param image: np.array, the raw image with shape like(height, width, n_classes)
    :param prob: np.array, same shape as `image`, giving the probabilities
    :param n_steps: int, number of iterations for CRF inference.
    :return:
        result: np.array(dtype=np.int32), result after the CRF post-processing.
    """
    height, width, n_classes = prob.shape
    d = DenseCRF2D(width, height, n_classes)

    # unary potential
    unary = unary_from_softmax(prob.transpose((2, 0, 1)))
    d.setUnaryEnergy(unary)

    # pairwise potential
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=2)

    # inference
    Q = d.inference(n_steps)
    result = np.argmax(Q, axis=0).reshape((height, width))
    return result
