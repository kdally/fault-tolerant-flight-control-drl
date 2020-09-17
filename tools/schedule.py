import math

def schedule(initial_value, second_value):
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    # def func(progress):
    #     """
    #     Progress will decrease from 1 (beginning) to 0
    #     :param progress: (float)
    #     :return: (float)
    #     """
    #     k = 2.1
    #     return initial_value * math.exp(-k * progress)

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        if progress < 0.5:
            return progress * initial_value
        else:
            return progress * second_value

    return func
