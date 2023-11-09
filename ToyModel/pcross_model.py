""" This module provides a set of functions to get pcross using a toy model of P3D Arinyo-i-Prats """

import numpy as np
import os, sys
import astropy.io.fits
from astropy.table import Table
import scipy
import yaml
import multiprocessing
from multiprocessing import Pool

sys.path.insert(0, os.environ['HOME']+'/Software/LyaP3D')
from tools import SPEED_LIGHT, LAMBDA_LYA
from truth_p3d_computation import compute_pcross_truth


def get_p_k_linear(k_pivot, A_alpha, n_alpha):
    """ At the CMB pivot scale: k_pivot=0.05 Mpc^-1, k_array in h/Mpc """
    h = 0.7
    # k_pivot = 0.05 / h # h Mpc^-1 # this value of k_pivot corresponds to CMB scale so it's normal to get max corr between A and n at this scale
    # k_pivot = 15 / h # h Mpc^-1
    k_max=100
    k_array = np.logspace(-5, np.log10(k_max), num=1000) # h Mpc^-1

    p_linear = A_alpha * (k_array / k_pivot)**n_alpha
    if np.any(p_linear<0):
        print('negative p_linear')
    
    p_k_linear = [k_array, p_linear]
    
    return p_k_linear


def get_p1d(k_par: np.array, k_pivot: float, A_alpha: float, n_alpha: float):
    """ 
    Args:
    
    errors: String, Default: 'no errors'
            Options: - 'no errors', 'add errors'
            
    covariance: String, Default: 'no covariance'
            Options: - 'no covariance', 'add covariance'
    
    """

    p_k_linear = get_p_k_linear(k_pivot=k_pivot, A_alpha=A_alpha, n_alpha=n_alpha)

    p1d = compute_pcross_truth(k_par=k_par, k_max=100, ang_sep=0, p_k_linear=p_k_linear, 
                               q1=0.666, q2=0, 
                               kv=0.935003735664152, a_v=0.561, b_v=1.58, 
                               k_p=13.5, a_p=2.0,  
                               b_delta_squared=0.012462846812427325, beta=1.385, 
                               model='model2')

    if np.any(p1d<0):
        print('negative p1d')

    return p1d


def get_p1d_all_params(k_par, k_pivot, A_alpha, n_alpha, q1, q2, kv, a_v, b_v, k_p, a_p, b_delta_squared, beta):
    """ 
    Args:
    
    errors: String, Default: 'no errors'
            Options: - 'no errors', 'add errors'
            
    covariance: String, Default: 'no covariance'
            Options: - 'no covariance', 'add covariance'
    
    """

    p_k_linear = get_p_k_linear(k_pivot=k_pivot, A_alpha=A_alpha, n_alpha=n_alpha)

    p1d = compute_pcross_truth(k_par=k_par, k_max=100, ang_sep=0, p_k_linear=p_k_linear, 
                               q1=q1, q2=q2, 
                               kv=kv, a_v=a_v, b_v=b_v, 
                               k_p=k_p, a_p=a_p,  
                               b_delta_squared=b_delta_squared, beta=beta, 
                               model='model2')

    if np.any(p1d<0):
        print('negative p1d')

    return p1d



    



def read_params(path_to_yaml_params_file):
    
    with open(path_to_yaml_params_file) as f:
        yaml_dict = yaml.safe_load(f)
        
    params_centers = dict()
    varying_params_names = []
    varying_params_range = dict()
    
    for param_name in yaml_dict.keys():
        parinfo = yaml_dict[param_name]
        params_centers[param_name] = parinfo['value']
        if parinfo['fix'] == False:
            varying_params_names.append(param_name)
            varying_params_range[param_name] = parinfo['limit']

    return params_centers, varying_params_names, varying_params_range


def get_likelihood_p_cross_varying_one_param(path_to_yaml_params_file, varying_param_name, k_par, k_max, ang_sep):
    """ This function is useless """

    # Reading yaml file
    params_centers, _, varying_params_range = read_params(path_to_yaml_params_file)
    
    # Defining params centers used for p_cross_center computation
    A_alpha = params_centers['A_alpha']
    n_alpha = params_centers['n_alpha']
    q1 = params_centers['q1']
    q2 = params_centers['q2']
    kv = params_centers['kv']
    a_v = params_centers['a_v']
    b_v = params_centers['b_v']
    k_p = params_centers['k_p']
    a_p = params_centers['a_p']
    b_delta_squared = params_centers['b_delta_squared']
    beta = params_centers['beta']
    
    # p_cross_center computation
    p_cross_center = get_p_cross(k_par=k_par, k_max=k_max, ang_sep=ang_sep, A_alpha=A_alpha, n_alpha=n_alpha, q1=q1, q2=q2, 
                                 kv=kv, a_v=a_v, b_v=b_v, 
                                 k_p=k_p, a_p=a_p,  
                                 b_delta_squared=b_delta_squared, beta=beta)
    
    # Defining varying param array from limits in yaml file
    varying_param_array = np.linspace(varying_params_range[varying_param_name][0], varying_params_range[varying_param_name][1], 5)
    
    # Initializing likelihood
    likelihood_p_cross_varying_one_param = 0
    
    # Varying param and computing likelihood at each time
    for j_varying_param_value, varying_param_value in enumerate(varying_param_array):
        
        # Changing the value of the param in params_centers used for p_cross_varying_param computation
        params_centers[varying_param_name] = varying_param_value
                
        A_alpha = params_centers['A_alpha']
        n_alpha = params_centers['n_alpha']
        q1 = params_centers['q1']
        q2 = params_centers['q2']
        kv = params_centers['kv']
        a_v = params_centers['a_v']
        b_v = params_centers['b_v']
        k_p = params_centers['k_p']
        a_p = params_centers['a_p']
        b_delta_squared = params_centers['b_delta_squared']
        beta = params_centers['beta']

        # Computing p_cross_varying_param
        p_cross_varying_param = get_p_cross(k_par=k_par, k_max=k_max, ang_sep=ang_sep, A_alpha=A_alpha, n_alpha=n_alpha, q1=q1, q2=q2, 
                                 kv=kv, a_v=a_v, b_v=b_v, 
                                 k_p=k_p, a_p=a_p,  
                                 b_delta_squared=b_delta_squared, beta=beta)
        
        # Likelihood
        likelihood_p_cross_varying_one_param += (p_cross_center - p_cross_varying_param)**2 ## TODO: implement errors and covariance
        
    return likelihood_p_cross_varying_one_param
        

def likelihood_p_cross_varying_params(path_to_yaml_params_file, k_par, k_max, ang_sep, ncpu):
    """ This function is useless """
    
    # Reading yaml file just to get varying_params_names
    _, varying_params_names, _ = read_params(path_to_yaml_params_file)
    
    # Parallelizing over the varying params
    with Pool(ncpu) as pool:
        likelihood_p_cross = pool.starmap(
            get_likelihood_p_cross_varying_one_param, [
                [path_to_yaml_params_file, varying_param_name, k_par, k_max, ang_sep] 
                for i_varying_param, varying_param in enumerate(varying_params_names)])
        
    # Adding all likelihoods
    likelihood_p_cross_total = 0
    likelihood_p_cross_total += [likelihood_p_cross[i] for i in range(len(likelihood_p_cross))]
    
    return likelihood_p_cross_total

