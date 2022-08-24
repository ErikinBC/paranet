"""
Use of numerical methods to solve for integrals
"""

import numpy as np
from scipy.integrate import simpson

def get_intergral(ff:np.ndarray, xseq:np.ndarray, method:str='trapezoidal') -> np.ndarray:
    """
    Calculate the integral for some 1_d function fun(x) over x

    Inputs
    ------
    ff:             Function evaluated at f(x)
    xseq:           Range of f(x) to evaluate over
    method:         Numerical intergration method (default="trapezoidal")

    Returns
    -------
    A float of the corresponding integral value
    """
    # Input checks
    lst_method = ['trapezoidal', 'simpson']
    assert method in lst_method, f"method must be one of: {', '.join(lst_method)}"
    # Calculate integral
    if method == 'trapezoidal':  # Trapezoidal rule
        # Sum over the axes
        deps = np.diff(xseq,axis=0)
        integral = np.sum(deps*(ff[1:]+ff[:-1])/2, axis=0)
    if method == 'simpson':  # Simpson's rule
        # Need to transpose to [k, f(x)]
        integral = simpson(y=ff.T, x=xseq.T)
    return integral
    
   