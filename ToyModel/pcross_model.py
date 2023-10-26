""" This module provides a set of functions to get pcross using a toy model of P3D Arinyo-i-Prats """

import numpy as np
import os, sys
import astropy.io.fits
from astropy.table import Table
import scipy


def p_linear(k_array, k_pivot, A_alpha, n_alpha):
    
    p_linear = A_alpha * (k / k_pivot)**n_alpha
    
    return p_linear


