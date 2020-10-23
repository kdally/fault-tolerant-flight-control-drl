import numpy as np


class NormalActionNoise:
    """
    A Gaussian action noise
    :param mean: (float) the mean value of the noise
    :param sigma: (float) the scale of the noise (std here)
    """

    def __init__(self, mean, sigma):
        super().__init__()
        self._mu = mean
        self._sigma = sigma

    def reset(self) -> None:
        """
        call end of episode reset for the noise
        """
        pass

    def __call__(self) -> np.ndarray:
        return np.random.normal(self._mu, self._sigma)