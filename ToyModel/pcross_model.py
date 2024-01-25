""" This module provides a set of functions to get pcross using a toy model of P3D Arinyo-i-Prats """

import numpy as np
import os, sys
import astropy.io.fits
from astropy.table import Table
import scipy
import yaml
import multiprocessing
from multiprocessing import Pool
import matplotlib.pyplot as plt

sys.path.insert(0, os.environ['HOME']+'/Software/LyaP3D')
from tools import SPEED_LIGHT, LAMBDA_LYA
from truth_p3d_computation import compute_pcross_truth

from iminuit.util import describe

import inspect

def get_A_alpha(k_pivot):
    """ Function that computes A_alpha for a fixed k_pivot"""
    h=0.7
    k_pivot_ref = 0.7 / h # h Mpc^-1 # this value of k_pivot corresponds to Pedersen
    A_alpha_ref = 0.35*(2*np.pi**2)/k_pivot_ref**3
    n_alpha_ref = -2.30
    ### Using the P_linear model for this Toy model
    A_alpha = A_alpha_ref * (k_pivot / k_pivot_ref)**n_alpha_ref
    print("k_pivot", k_pivot)
    print("A_alpha", A_alpha)
    return A_alpha


def get_p_k_linear(k_pivot, A_alpha, n_alpha):
    """ At the CMB pivot scale: k_pivot=0.05 Mpc^-1, k_array in h/Mpc """
    h = 0.7
    # k_pivot = 0.05 / h # h Mpc^-1 # this value of k_pivot corresponds to CMB scale so it's normal to get max corr between A and n at this scale
    # k_pivot = 15 / h # h Mpc^-1
    k_max = 100
    k_array = np.logspace(-5, np.log10(k_max), num=1000) # h Mpc^-1

    p_linear = A_alpha * (k_array / k_pivot)**n_alpha
    if np.any(p_linear<0):
        print('negative p_linear')
    
    p_k_linear = [k_array, p_linear]

    return p_k_linear


def get_p_k_linear_wdm(k_pivot, A_alpha, n_alpha, alpha):
    """ Including WDM model, beta and gamma being fixed, alpha varying, k_array in h/Mpc """
    h = 0.7
    k_max = 100
    k_array = np.logspace(-5, np.log10(k_max), num=1000) # h Mpc^-1

    # WDM params
    nu = 1.12
    beta = 2 * nu
    gamma = -5 / nu
    T = (1 + (alpha * k_array)**beta)**gamma

    # p_linear
    p_linear = A_alpha * (k_array / k_pivot)**n_alpha

    # p_linear WDM
    p_linear_WDM = p_linear * (T**2)

    if np.any(p_linear<0):
        print('negative p_linear')

    p_k_linear = [k_array, p_linear_WDM]

    return p_k_linear


def get_p1d(k_par, k_pivot, A_alpha, n_alpha):
    """ 
    Function used for p1d minimization
    - k_max and ang_sep nare fixed, k_pivot will be fixed also
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
    Function used for p1d minimization with the option to vary all params
    - k_max and ang_sep nare fixed, k_pivot will be fixed also
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


def get_p1d_all_params_wdm(k_par, k_pivot, A_alpha, n_alpha, alpha, q1, q2, kv, a_v, b_v, k_p, a_p, b_delta_squared, beta):
    """
    Function used for p1d minimization with the option to vary all params
    - k_max and ang_sep nare fixed, k_pivot will be fixed also
    """

    p_k_linear = get_p_k_linear(k_pivot=k_pivot, A_alpha=A_alpha, n_alpha=n_alpha, alpha=alpha)

    p1d = compute_pcross_truth(k_par=k_par, k_max=100, ang_sep=0, p_k_linear=p_k_linear, 
                               q1=q1, q2=q2, 
                               kv=kv, a_v=a_v, b_v=b_v, 
                               k_p=k_p, a_p=a_p,  
                               b_delta_squared=b_delta_squared, beta=beta, 
                               model='model2')

    if np.any(p1d<0):
        print('negative p1d')

    return p1d


def get_pcross_all_params(k_par, k_pivot, A_alpha, n_alpha, q1, q2, kv, a_v, b_v, k_p, a_p, b_delta_squared, beta):
    """ 
    Function used for pcross minimization with the option to vary all params
    - k_max and ang_sep are fixed, k_pivot will be fixed also
    """

    ang_sep_Mpc_h = np.linspace(0, 15, 10)

    p_k_linear = get_p_k_linear(k_pivot=k_pivot, A_alpha=A_alpha, n_alpha=n_alpha)

    pcross = compute_pcross_truth(k_par=k_par, k_max=100, ang_sep=ang_sep_Mpc_h, p_k_linear=p_k_linear, 
                               q1=q1, q2=q2, 
                               kv=kv, a_v=a_v, b_v=b_v, 
                               k_p=k_p, a_p=a_p,  
                               b_delta_squared=b_delta_squared, beta=beta, 
                               model='model2')

    return pcross


def get_pcross_all_params_wdm(k_par, k_pivot, A_alpha, n_alpha, alpha, q1, q2, kv, a_v, b_v, k_p, a_p, b_delta_squared, beta):
    """ 
    Function used for pcross minimization with the option to vary all params
    - k_max and ang_sep are fixed, k_pivot will be fixed also
    """

    ang_sep_Mpc_h = np.linspace(0, 15, 10)

    p_k_linear = get_p_k_linear(k_pivot=k_pivot, A_alpha=A_alpha, n_alpha=n_alpha, alpha=alpha)

    pcross = compute_pcross_truth(k_par=k_par, k_max=100, ang_sep=ang_sep_Mpc_h, p_k_linear=p_k_linear, 
                               q1=q1, q2=q2, 
                               kv=kv, a_v=a_v, b_v=b_v, 
                               k_p=k_p, a_p=a_p,  
                               b_delta_squared=b_delta_squared, beta=beta, 
                               model='model2')

    return pcross


def vary_params(minuit_object, varying_params_keys):
    """ This function fixes first all params, then varies the ones in varying_params_keys list """

    minuit_object.fixed = True

    for keys in varying_params_keys:
        minuit_object.fixed[keys] = False
        
    return minuit_object


def get_function_args(function):
    # Get the signature of the function
    signature = inspect.signature(function)

    # Access the parameters of the function
    parameters = signature.parameters

    # Extract the parameter names
    argument_list = list(parameters.keys())[1:]

    print(argument_list)
    
    return argument_list


class Likelihood_Pcross:
    """ Likelihood cost function used in minuit minimization
    - model is a function that predicts y for given x (ex: get_p1d_all_params or get_pcross_all_params)
    - x is the k_par_array
    - y is the p1d_data
    - err is the error_p1d_data
    - minimize_chi_square is a boolean, when true the function minimized is the chi_square and not the likelihood
    - ym is the p1d_model
    """

    # errordef = Minuit.LEAST_SQUARES  # for Minuit to compute errors correctly

    def __init__(self, model, x, y, err, minimize_chi_square, verbose: int = 0):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.err = np.asarray(err)
        self.model = model
        self.minimize_chi_square = minimize_chi_square
        self.verbose = verbose

        params = describe(self.model)[1:]
        # Might be removed later
        if params[0]=='z':
            params = describe(self.model)[2:]
        ##
        print('params', params)
        print(*params)
        self.parameters = params
        # print(self.parameters)
        # params = get_function_args(self.model)
        # print(params)
        # print(*params)
        # self.parameters = params

    def __call__(self, *params):  # we must accept a variable number of model parameters
    # def _call(self, params):
        ym = self.model(self.x, *params)

        chi_square = np.sum((self.y - ym) ** 2 / self.err ** 2)

        if self.minimize_chi_square is True:
            return chi_square
        else:
            likelihood = -0.5 * chi_square
            return likelihood


class Likelihood_Pcross_test:
    """ Likelihood cost function used in minuit minimization
    - model is a function that predicts y for given x (ex: get_p1d_all_params or get_pcross_all_params)
    - x is the k_par_array
    - y is the p1d_data
    - err is the error_p1d_data
    - minimize_chi_square is a boolean, when true the function minimized is the chi_square and not the likelihood
    - ym is the p1d_model
    """

    # errordef = Minuit.LEAST_SQUARES  # for Minuit to compute errors correctly

    def __init__(self, model, x, y, err, minimize_chi_square, verbose: int = 0):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.err = np.asarray(err)
        self.model = model
        self.minimize_chi_square = minimize_chi_square
        self.verbose = verbose

        params = get_function_args(self.model)
        self.parameters = params
        print(params)
        print(*params)
        
    # def compute_chi_square(y, err, ym):
    #     chi_square = np.sum((y - ym) ** 2 / err ** 2)
    #     return chi_square

    def __call__(self, *params):  # we must accept a variable number of model parameters
        ym = self.model(self.x, *params)
        chi_square = np.sum((self.y - ym) ** 2 / self.err ** 2)

        if self.minimize_chi_square is True:
            self.chisquare = chi_square
            # self.chisquare = self.compute_chi_square(self.y, self.err, ym)
            self.likelihood = None
            return self
        else:
            self.chi_square = None
            self.likelihood = -0.5 * chi_square
        return self


class Likelihood_Pcross_test_z:
    """ Likelihood cost function used in minuit minimization
    - model is a function that predicts y for given x (ex: get_p1d_all_params or get_pcross_all_params)
    - x is the k_par_array
    - y is the p1d_data
    - err is the error_p1d_data
    - minimize_chi_square is a boolean, when true the function minimized is the chi_square and not the likelihood
    - ym is the p1d_model
    """

    # errordef = Minuit.LEAST_SQUARES  # for Minuit to compute errors correctly

    def __init__(self, model, x, y, err, z, minimize_chi_square):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.err = np.asarray(err)
        self.model = model
        self.minimize_chi_square = minimize_chi_square
        self.z = np.asarray(z)

        # params = describe(model)[1:]
        params = describe(model)[2:] #Because k_par and z aren't params to be fitted: Must be made with get_p1d_all_params_all_z_test
        self.parameters = params
        print(params)

    def __call__(self, *params):  # we must accept a variable number of model parameters
        ym = self.model(self.x, *params)

        chi_square = np.sum((self.y - ym) ** 2 / self.err ** 2)

        if self.minimize_chi_square is True:
            return chi_square
        else:
            likelihood = -0.5 * chi_square
            return likelihood


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

