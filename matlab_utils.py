"""
a few matlab utility functions, so far only to do with matlab and numpy using different defintions for arrays
"""

import matlab as mat
import numpy as np

def np2mat(a):
    if type(a)==list:
        return mat.double(a) #only because sometimes they can get mixed up
    else: #assume an np array
        return mat.double(a.tolist())
def mat2np(a):
    return np.array(a)
