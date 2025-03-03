""" This module provides a set of functions to get a prediction of Px using ForestFlow"""

import numpy as np
import os, sys
sys.path.insert(0, os.environ['HOME']+'/Software/LyaP3D')
from tools import SPEED_LIGHT, LAMBDA_LYA

# For forestflow
from forestflow.model_p3d_arinyo import ArinyoModel
from forestflow.archive import GadgetArchive3D
from forestflow.P3D_cINN import P3DEmulator
# convert angular coordinates to physical
from astropy.cosmology import FlatLambdaCDM
from forestflow import pcross
import forestflow
from lace.cosmo.thermal_broadening import thermal_broadening_kms
from lace.cosmo import camb_cosmo

from functools import partial


def get_pcross_forestflow_old(z_target, k_parallel, transverse_sep, mF, gamma, sigT_Mpc, kF_Mpc, Archive3D=None, 
                          k_parallel_units='I_Angstrom', transverse_sep_units='deg'):
    """ Old function to be removed """
    if k_parallel[0] == 0:
        sys.exit('k_parallel array must not have a zero')

    path_program = forestflow.__path__[0][:-10]

    # input emu
    z = z_target

    ## Hardcoded cosmo for now
    omnuh2 = 0.0006
    mnu = omnuh2 * 93.14
    H0 = 67.36
    omch2 = 0.12
    ombh2 = 0.02237
    As = 2.1e-9
    ns = 0.9649
    nrun = 0.0
    w = -1.0
    omk = 0
    cosmo = {
        'H0': H0,
        'omch2': omch2,
        'ombh2': ombh2,
        'mnu': mnu,
        'omk': omk,
        'As': As,
        'ns': ns,
        'nrun': nrun,
        'w': w
    }
    sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(cosmo)

    # Conversions needed for forestflow (It takes all in Mpc units)
    if k_parallel_units == 'I_kmps':
        dkms_dMpc_zs = camb_cosmo.dkms_dMpc(sim_cosmo, z=np.array([z]))
        kpar_iMpc = k_parallel * dkms_dMpc_zs
    elif k_parallel_units == 'I_Angstrom':
        dA_dMpc_zs = (astropy_cosmo.H(z).value * LAMBDA_LYA) / SPEED_LIGHT
        kpar_iMpc = k_parallel * dA_dMpc_zs
    elif k_parallel_units == 'I_Mpc':
        kpar_iMpc = k_parallel
    else:
        sys.exit('Must input k_parallel_units: I_Mpc/I_kmps/I_Angstrom')

    if transverse_sep_units == 'deg':
        h = H0 / 100
        Om0 = (omch2 + ombh2) / (h**2)
        astropy_cosmo = FlatLambdaCDM(H0=H0, Om0=Om0, m_nu=[mnu, 0, 0])
        sepbins_cMpc = astropy_cosmo.comoving_transverse_distance(z) * np.deg2rad(sepbins_deg)
    elif transverse_sep_units == 'Mpc':
        sepbins_cMpc = transverse_sep
    else:
        sys.exit('Must input transverse_sep_units: Mpc or deg')

    # emulator parameters
    emu_params = {
    "mF": mF,
    "gamma": gamma,
    "sigT_Mpc":sigT_Mpc,
    "kF_Mpc":kF_Mpc,}

    # LOAD P3D ARCHIVE
    if Archive3D is None:
        folder_lya_data = path_program + "/data/best_arinyo/"
        folder_interp = path_program + "/data/plin_interp/"
        
        Archive3D = GadgetArchive3D(
            base_folder=path_program[:-1],
            folder_data=folder_lya_data,
            force_recompute_plin=False,
            average="both",
        )

    # Load emulator
    training_type = "Arinyo_min"
    model_path=path_program+"/data/emulator_models/mpg_hypercube.pt"
    
    emulator = P3DEmulator(
        Archive3D.training_data,
        Archive3D.emu_params,
        nepochs=300,
        lr=0.001,  # 0.005
        batch_size=20,
        step_size=200,
        gamma=0.1,
        weight_decay=0,
        adamw=True,
        nLayers_inn=12,  # 15
        Archive=Archive3D,
        Nrealizations=10000,
        training_type=training_type,
        model_path=model_path,
    )

    info_power = {
    "cosmo": cosmo,
    "z": z,
}

    out = emulator.evaluate(
        emu_params=emu_params,
        info_power=info_power,
        Nrealizations=10000
    )

    # now initialize the Arinyo model with the emulated values of bias and beta
    camb_results = camb_cosmo.get_camb_results(sim_cosmo, zs=[z], camb_kmax_Mpc=1000) # set default cosmo
    arinyo = ArinyoModel(cosmo=sim_cosmo, camb_results=camb_results, zs=[z], camb_kmax_Mpc=1000) # set model

    # Predict Px
    rperp_pred, Px_pred_Mpc = pcross.Px_Mpc_detailed(kpar_iMpc,
    arinyo.P3D_Mpc,
    z,
    rperp_choice=sepbins_cMpc,
    P3D_mode='pol',
    min_kperp=10**-3,
    max_kperp=10**2.9,
    nkperp=2**12,
    **{"pp":out['coeffs_Arinyo']})

    ## ADD option that converts output

    return rperp_pred, kpar_iMpc, Px_pred_Mpc.T


def load_emulator():
    """ This function loads the emulator and doesn't require any input """

    path_program = forestflow.__path__[0][:-10]

    # LOAD P3D ARCHIVE
    folder_lya_data = path_program + "/data/best_arinyo/"
    folder_interp = path_program + "/data/plin_interp/"

    Archive3D = GadgetArchive3D(
        base_folder=path_program[:-1],
        folder_data=folder_lya_data,
        force_recompute_plin=False,
        average="both",
    )

    # Load emulator
    training_type = "Arinyo_min"
    model_path=path_program+"/data/emulator_models/mpg_hypercube.pt"

    emulator = P3DEmulator(
        Archive3D.training_data,
        Archive3D.emu_params,
        nepochs=300,
        lr=0.001,  # 0.005
        batch_size=20,
        step_size=200,
        gamma=0.1,
        weight_decay=0,
        adamw=True,
        nLayers_inn=12,  # 15
        Archive=Archive3D,
        training_type=training_type,
        model_path=model_path,
    )

    return emulator


def load_arinyo(z, cosmo_param_dict):
    """ This function reads redshift z and cosmo paremeters dictionary and loads the Arinyo model
    Arguments:
    ----------
    z: Float or array of floats
    Redshift.

    cosmo_param_dict: Dictionary
    It should include 'H0', 'omch2', 'ombh2', 'mnu', 'omk', 'As', 'ns', 'nrun', 'w'.

    Return:
    -------
    arinyo: ??
    """

    # Getting sim_cosmo from camb_cosmo based on cosmo input
    sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(cosmo_param_dict)

    # Initializing the Arinyo model with the emulated values of bias and beta
    camb_results = camb_cosmo.get_camb_results(sim_cosmo, zs=[z], camb_kmax_Mpc=1000) # set default cosmo
    arinyo = ArinyoModel(cosmo=sim_cosmo, camb_results=camb_results, zs=[z], camb_kmax_Mpc=1000) # set model

    return arinyo


def get_forestflow_params(z, igm_param_dict, cosmo_param_dict, sim_cosmo, dkms_dMpc_zs):
    """ This function: 
        - Reads the IGM parameters from igm_param_dict
        - Reads the cosmo parameters from cosmo_param_dict
        - Converts IGM parameters T0 and lambda_pressure to emulator parameters sigT_Mpc and kF_Mpc respectively
        - Defines emualator parameters dictionary emu_params and linear power spectrum information dictionary 
        containing redshift and cosmo parameters.
        - Returns them

    Arguments:
    ----------
    z: Float or array of floats
    Redshift.

    igm_param_dict: Dictionary 
    It should include 'T0' [K], 'gamma', 'mF', 'lambda_pressure' [kpc].

    cosmo_param_dict: Dictionary
    It should include 'H0', 'omch2', 'ombh2', 'mnu', 'omk', 'As', 'ns', 'nrun', 'w'.

    Return:
    -------
    emu_params: Dictionary
    It includes the emulator parameters: 'mF', 'gamma', 'sigT_Mpc', 'kF_Mpc'.

    info_power: Dictionary
    It includes redshift 'z' and cosmo parameters 'cosmo'.
    """

    # Getting sim_cosmo from camb_cosmo based on cosmo input to be used for conversions
    # sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(cosmo_param_dict)

    # Converting IGM parameters T0 and lambda_pressure to emulator parameters sigT_Mpc and kF_Mpc respectively
    sigT_kms = thermal_broadening_kms(igm_param_dict['T0'])
    # dkms_dMpc_zs = camb_cosmo.dkms_dMpc(sim_cosmo, z=np.array([z]))
    sigT_Mpc = sigT_kms / dkms_dMpc_zs[0]
    kF_Mpc = 1 / (igm_param_dict['lambda_pressure'] / 1000)

    # Defining dictionaries
    emu_params = {
        "mF": igm_param_dict['mF'],
        "gamma": igm_param_dict['gamma'],
        "sigT_Mpc":sigT_Mpc,
        "kF_Mpc":kF_Mpc,
    }

    info_power = {
    "cosmo": cosmo_param_dict,
    "z": z,
    }
    return emu_params, info_power


def convert_kpar_to_forestflow_units(z, kpar, inout_unit, dAA_dMpc_zs=None, dkms_dMpc_zs=None):
    """ Converts kpar to Mpc, the only input unit accepted by forestflow """

    if inout_unit == 'AA':
        if dAA_dMpc_zs is not None:
            kpar_iMpc = kpar * dAA_dMpc_zs
        else:
            sys.exit('Must input dAA_dMpc_zs')
    elif inout_unit == 'kmps':
        if dkms_dMpc_zs is not None:
            kpar_iMpc = kpar * dkms_dMpc_zs
        else:
            sys.exit('Must input dkms_dMpc_zs') 
    else:
        sys.exit('Must input inout_unit: AA or kmps')

    return kpar_iMpc


def convert_pcross_to_output_units(z, Px_pred_Mpc, inout_unit, dAA_dMpc_zs=None, dkms_dMpc_zs=None):
    """ Converts the output of forestflow from Mpc to the desired output unit """

    if inout_unit == 'AA':
        if dAA_dMpc_zs is not None:
            Px_pred_outunits = Px_pred_Mpc / dAA_dMpc_zs
        else:
            sys.exit('Must input dAA_dMpc_zs')
    elif inout_unit == 'kmps':
        if dkms_dMpc_zs is not None:
            Px_pred_outunits = Px_pred_Mpc / dkms_dMpc_zs
        else:
            sys.exit('Must input dkms_dMpc_zs') 
    else:
        sys.exit('Must input inout_unit: AA or kmps')

    return Px_pred_outunits


def convert_sepbins_to_foresflow_units(z, cosmo_param_dict, sepbins, sepbins_unit):
    """ Converts the transverse separation values from their input unit to Mpc as accepted by forestflow """

    H0 = cosmo_param_dict['H0']
    omch2 = cosmo_param_dict['omch2']
    ombh2 = cosmo_param_dict['ombh2']
    mnu = cosmo_param_dict['mnu']

    if sepbins_unit == 'deg':
        h = H0 / 100
        Om0 = (omch2 + ombh2) / (h**2)
        astropy_cosmo = FlatLambdaCDM(H0=H0, Om0=Om0, m_nu=[mnu, 0, 0])
        sepbins_Mpc = astropy_cosmo.comoving_transverse_distance(z) * np.deg2rad(sepbins)
    elif sepbins_unit == 'Mpc':
        sepbins_Mpc = sepbins
    else:
        sys.exit('Must input sepbins_unit: Mpc or deg')

    return sepbins_Mpc


def get_pcross_forestflow(kpar, sepbins, z, cosmo_param_dict, sim_cosmo, dAA_dMpc_zs, dkms_dMpc_zs, 
                          emulator, arinyo, inout_unit, sepbins_unit, mF, T0, gamma, lambda_pressure):
    """ This function predicts Px from forestflow given the IGM and cosmo parameters of the input.
    PS: the function is not yet adapted to vary cosmology.

    Arguments:
    ----------
    z: Float or array of floats
    Redshift.

    mF: Float or array of floats []
    Mean transmitted flux fraction. It is just an array if z is an array.

    T0: Float or array of floats
    Amplitude of the temperature density relation T = T0 * delta_b**(gamma - 1). It is just an array if z is an array.

    gamma: Float or array of floats
    Slope of the temperature density relation T = T0 * delta_b**(gamma - 1). It is just an array if z is an array.

    lambda_pressure: Float or array of floats
    Pressure smoothing scale (Jeans smoothing): The scale where pressure overcomes gravity at small scales 
    -> smoothing of fluctuations.

    cosmo_param_dict: Dictionary
    Dictionary of cosmo parameters. It should include 'H0', 'omch2', 'ombh2', 'mnu', 'omk', 'As', 'ns', 'nrun', 'w'. 
    PS: they vary as function of redshift z, but are given as input to this function since cosmo is not to be varied for the moment.

    emulator: Emulator already loaded using load_emulator() function.

    arinyo: Loaded using load_arinyo() function. It must be given as input as long as the cosmo is not to be varied for now.

    kpar: Array of floats
    Array of k_parallel at which we want to get a prediction.

    sepbins: Array of floats
    Array of sepbins at which we want to get a prediction.

    inout_unit: String, default: 'AA', options: 'kmps'
    Units of input kpar that must be given in terms of the output units we want, and the output will be given in that same unit.

    sepbins_unit: Sting, default: 'deg', options: 'Mpc'.
    Units of separation values at which we want to get the prediction.

    Return:
    -------
    Px_pred_output_units: 
    
    """

    # Code won't work if kpar has a zero
    if kpar[0] == 0:
        sys.exit('kpar array must not have a zero')

    # Conversions made at the beginning so that if the units are wrong, the code exists from here
    kpar_iMpc = convert_kpar_to_forestflow_units(z, kpar, inout_unit, dAA_dMpc_zs=dAA_dMpc_zs, dkms_dMpc_zs=dkms_dMpc_zs)
    # print('Converted kpar to Mpc^-1 units:', kpar_iMpc)
    sepbins_Mpc = convert_sepbins_to_foresflow_units(z, cosmo_param_dict, sepbins, sepbins_unit)
    # print('Converted sepbins to Mpc units:', sepbins_Mpc)

    # Creating dictionary of IGM parameters that will then be transformed into emu_params
    # PS: cosmo parameters are not given explicitely since they're not varied for now
    igm_param_dict = {
        "mF": mF,
        "gamma": gamma,
        "lambda_pressure": lambda_pressure,
        "T0": T0,
    }

    # Getting forestflow parameters
    emu_params, info_power = get_forestflow_params(z, igm_param_dict, cosmo_param_dict, sim_cosmo, dkms_dMpc_zs)
    # print('Input parameters given to the emulator are:', emu_params, info_power)

    # Evaluating emulator at the input parameters values
    out = emulator.evaluate(
        emu_params=emu_params,
        info_power=info_power)

    # Predict Px
    rperp_pred, Px_pred_Mpc = pcross.Px_Mpc_detailed(kpar_iMpc,
    arinyo.P3D_Mpc,
    info_power['z'],
    rperp_choice=sepbins_Mpc,
    P3D_mode='pol',
    min_kperp=10**-3,
    max_kperp=10**2.9,
    nkperp=2**12,
    **{"pp":out['coeffs_Arinyo']})

    # Convert Px_pred_Mpc to Px_pred_output that has inout_units
    Px_pred_output = convert_pcross_to_output_units(z, Px_pred_Mpc, inout_unit, dAA_dMpc_zs=dAA_dMpc_zs, dkms_dMpc_zs=dkms_dMpc_zs)

    # Return transpose to match Px_data shapes
    Px_pred_output_transpose = Px_pred_output.T

    return Px_pred_output_transpose

