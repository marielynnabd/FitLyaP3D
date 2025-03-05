""" This module provides a set of function to cumpute the likelihood or chi_square """
import numpy as np
from iminuit.util import describe


class Likelihood_Pcross_oneZ:
    """ Obsolete """
    """ Likelihood cost function used in minuit minimization
    - model is a function that predicts y for given x (ex: get_p1d_all_params or get_pcross_all_params)
    - x is the k_par_array
    - y is the p1d_data
    - err is the error_p1d_data
    - minimize_chi_square is a boolean, when true the function minimized is the chi_square and not the likelihood
    - ym is the p1d_model
    """

    # errordef = Minuit.LEAST_SQUARES  # for Minuit to compute errors correctly

    def __init__(self, model, x, y, err, minimize_chi_square,
                 kpar, sepbins, z, cosmo_param_dict, sim_cosmo, emulator, arinyo, inout_unit, sepbins_unit, 
                 verbose: int = 0):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.err = np.asarray(err)
        self.model = model
        self.minimize_chi_square = minimize_chi_square
        # self.kpar = kpar # no need for it since it will be the same as self.x
        self.sepbins = np.asarray(sepbins)
        self.redshift = z
        self.cosmo = cosmo_param_dict
        self.sim_cosmo = sim_cosmo
        self.emulator = emulator
        self.arinyo = arinyo
        self.inout_unit = inout_unit
        self.sepbins_unit = sepbins_unit
        self.verbose = verbose

        # params = describe(self.model)
        params = describe(self.model)[9:] # Must be improved
        # params = describe(self.model)[1:]
        # # Might be removed later
        # if params[0]=='z':
        #     params = describe(self.model)[2:]
        ##
        print('params', params)
        print(*params)
        # self.parameters = params
        # print(self.parameters)
        # params = get_function_args(self.model)
        # print(params)
        # print(*params)
        # self.parameters = params

    def __call__(self, *params):  # we must accept a variable number of model parameters
    # def _call(self, params):
        ym = self.model(self.x, self.sepbins, self.redshift, self.cosmo, self.sim_cosmo,
                          self.emulator, self.arinyo, self.inout_unit, self.sepbins_unit, *params)

        chi_square = np.sum((self.y - ym) ** 2 / self.err ** 2)

        if self.minimize_chi_square is True:
            return chi_square
        else:
            likelihood = -0.5 * chi_square
            return likelihood


def vary_params(minuit_object, varying_params_keys):
    """ This function fixes first all params, then varies the ones in varying_params_keys list """

    minuit_object.fixed = True

    for keys in varying_params_keys:
        minuit_object.fixed[keys] = False
        
    return minuit_object


class Likelihood_Pcross:
    """ Likelihood cost function used in Minuit minimization """

    def __init__(self, model, y_data, y_err, minimize_chi_square=True, return_model=False, verbose=0):
        """
        model: function that takes (mF, T0, gamma, lambda_pressure) and returns predicted Px
        y_data: measured Px values
        y_err: errors on Px
        return_model: this must be false when running Iminuit
        """
        self.model = model  # The wrapped model function
        self.y_data = np.asarray(y_data)
        self.y_err = np.asarray(y_err)
        self.minimize_chi_square = minimize_chi_square
        self.return_model = return_model
        self.verbose = verbose

        # Extract parameter names automatically
        self.parameters = describe(model)  # Only free parameters remain
        print(self.parameters)
        print(self.model)

    def __call__(self, mF, T0, gamma, lambda_pressure):
        """ Computes chi-square or log-likelihood given parameter values """

        # Compute model prediction with current parameters
        y_model = self.model(mF, T0, gamma, lambda_pressure)

        # Compute chi-square
        chi_square = np.sum((self.y_data - y_model) ** 2 / self.y_err ** 2)

        if self.minimize_chi_square:
            if self.return_model:
                return y_model, chi_square
            else:
                return chi_square
        else:
            if self.return_model:
                return y_model, -0.5 * chi_square  # Log-likelihood
            else:
                return  -0.5 * chi_square



