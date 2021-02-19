import numpy as np
import cv2 as cv

import numba
from numba import jit


@jit(nopython = True)
def imgContains(img,pt):
    return ( pt[0] >= 0 and pt[0] < img.shape[1] and pt[1] >= 0 and pt[1] < img.shape[0] )

