# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _citation as citation
else:
    import _citation as citation
import numpy as np

def initialize():
    return citation.initialize()


def terminate():
    return citation.terminate()


def step(cmd, env, failure):
    return citation.step(cmd, env, failure)

initialize()

print(step(np.zeros(10),np.zeros(9),0))

