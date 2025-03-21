""" This module provides a set of function to cumpute the likelihood or chi_square """
import numpy as np
from iminuit.util import describe


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
        if y_model is None:
            chi_square = np.inf
        else:
            chi_square = np.sum((self.y_data - y_model) ** 2 / self.y_err ** 2)
        print('Chi_square = ', chi_square)

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



