from lace.cosmo import camb_cosmo

# input your data
# import iminuit

# set your z 
z = 2.2
# set our cosmo (fixed)

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
# get the prediction for one z and many theta bins

# Loading emulator
emulator = load_emulator()

# Loading arinyo
sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(cosmo) # Put that inside the function of load arinyo
arinyo = load_arinyo(sim_cosmo, z)

# wrapper function:
# get_pcross_forestflow(params, that, we, might, vary, params, not, varied)

# likelihood_pcross = Likelihood_Pcross(get_pcross_forestflow(mF, T0, gamma, lambda_pressure, sepbins_Mpc, cosmo, emulator, arinyo)), k_par_array_hpMpc, pcross_data, error_pcross_data, minimize_chi_square=True)
