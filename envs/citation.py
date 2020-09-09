# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _citation as C_MODEL
else:
    import _citation as C_MODEL


def initialize():
    return C_MODEL.initialize()

def terminate():
    return C_MODEL.terminate()

def step(cmd):
    return C_MODEL.step(cmd)


initialize()

import numpy as np

initialize()
print(step(np.ones(10)))

print(step(np.ones(10)))

