"""
UTILITY FUNCTIONS
"""

# External modules
import os
import numpy as np
import pandas as pd
from typing import Callable

# List of currently supported distributions
dist_valid = ['exponential', 'weibull', 'gompertz']


# Set up the bounds for the different distributions (shape param is redundant for exponential)
di_bounds = {'exponential':((0, None), (0, None)),
             'weibull':((0, None), (0, None)),
             'gompertz':((None, None), (0, None))}


def check_d(d:np.ndarray) -> None:
    """Checks that the d array is only zeros or ones"""
    assert np.all((d == 1) | (d == 0)), 'Expected all values of d to be in {0,1}'


def check_t(t:np.ndarray) -> None:
    """Checks that the t array is strictly positive"""
    assert np.all(t > 0), 'Expected all values of t to be > 0'


def find_nearest_decimal(x:float) -> float:
    k = 0
    z = np.round(x, k)
    while z < 1:
        k += 1
        f = 10**k
        z = x * f    
    ub = np.ceil(z) / f
    if ub == x:
        ub = (np.ceil(z) + 1) / f
    return ub


def try_squeeze(x:np.ndarray, axis:int or None=None) -> np.ndarray:
    """Implements np.sqeeze but will not squeeze axis if length > 1"""
    if axis is not None:
        if x.shape[axis] == 1:
            return np.squeeze(x, axis)
        else:
            return x
    else:
        return np.squeeze(x, axis)


def close2zero(x:float or np.ndarray, tol:float=1e-10) -> None:
    """Check that the input is approximately zero"""
    return np.all(np.abs(x) < tol)


def should_fail(fun, **kwargs) -> None:
    """Test that a function evaluated for certain values fails"""
    try:
        failed = False
        fun(**kwargs)
    except:
        failed = True
    assert failed, 'Expected function to fail'


def all_or_None(lst:list or np.ndarray) -> bool:
    """Checks whether a list or array is either all None or not a single None"""
    is_none = [z is None for z in lst]
    not_none = [not z for z in is_none]
    check = all(is_none) or all(not_none)
    return check

def not_none(x) -> bool:
    """Checks that some input is not None"""
    return x is not None


def grad_finite_differences(f:Callable, x:np.ndarray, eps:float=1e-10, **args) -> np.ndarray:
    """
    Calculates the derivative for a function over a single parameter (first argument)

    Inputs
    ------
    f:              Some smooth function
    x:              Input to calculate derivative over: f'(x)
    eps:            Size of perturbation to calculate finite differences: (f(x+eps)-f(x-eps))/(2eps)
    **args:         All other arguments to pass into f(x, **args)

    Returns
    -------
    Will return a gradient the same size as the input array x
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    x_hi, x_lo = x + eps, x - eps
    f_hi, f_lo = f(x_hi, **args), f(x_lo, **args)
    grad = (f_hi - f_lo) / (2*eps)
    return grad


def gg_save(fn:str, fold:str, gg, width:float=5, height:float=4) -> None:
    """
    Wrapper to save a ggplot or patchworklib object object
    Inputs
    ------
    fn:         Filename to save (should end with .{png,jpg,etc})
    fold:       Folder to write the image to
    gg:         The plotnine ggplot object
    width:      Width of image (inches)
    height:     Height of image (inches)
    """
    gg_type = str(type(gg))  # Get the type
    path = os.path.join(fold, fn)  # Pre-set the path
    if os.path.exists(path):
        os.remove(path)  # Remove figure if it already exists
    if gg_type == "<class 'plotnine.ggplot.ggplot'>":
        gg.save(path, width=width, height=height, limitsize=False)
    elif gg_type == "<class 'patchworklib.patchworklib.Bricks'>":
        gg.savefig(fname=path)
    else:
        print('gg is not a recordnized type')


def check_interval(x:np.ndarray or float or pd.Series, low:int or float, high:int or float, equals_both:bool=False, equals_low:bool=False, equals_high:bool=False) -> None:
    """Check that an array or float is between low <= x <= high"""
    if equals_low and not equals_high:
        assert np.all((x >= low) & (x < high)), 'x is not between [low,high)'
    elif equals_high and not equals_low:
        assert np.all((x > low) & (x <= high)), 'x is not between [low,high)'
    elif equals_both or (equals_low and equals_high):
        assert np.all((x >= low) & (x <= high)), 'x is not between [low,high]'
    else:
        assert np.all((x > low) & (x < high)), 'x is not between (low,high)'


def str2lst(x) -> list:
    """Will convert a string to a list of length one"""
    if isinstance(x, str):
        x = [x]
    if isinstance(x, np.ndarray) or isinstance(x, pd.Series):
        x = list(x)
    return x


def dist2idx(dist:list) -> dict:
    """
    Returns a dictionary assigning the distributions to the different column indices
    """
    check_type(dist, list, 'dist')
    didx = {}
    for i, d in enumerate(dist):
        if d in didx:
            didx[d] = didx[d] + [i]
        else:
            didx[d] = [i]
    return didx


def broadcast_dist(dist:str or list, k:int) -> list:
    """
    If dist is a string or list of lenght one, will make into a list of length k if scale is an (1,k) array
    """
    dist = str2lst(dist)
    n_dist = len(dist)
    if k > n_dist:
        dist = dist * k
    return dist


def broadcast_td_dist(t:np.ndarray, d:np.ndarray, dist:str or list) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Broadcast either the time/censor vectors or the dist. For example, if t ~ (n,1) and len(dist)=k, then t ~ (n,k)
    """
    # Input checks
    assert hasattr(t, 'shape'), 't needs to be a np.ndarray'
    assert len(t.shape) == 2, 't needs to be an (n,k) array'
    # Do transforms
    dist = str2lst(dist)
    k_dist = len(dist)
    k_t = t.shape[1]
    if (k_dist == 1) & (k_t > 1):
        dist = broadcast_dist(dist, k_t)
    if (k_dist > 1) & (k_t == 1):
        t = np.tile(t, [1,k_dist])
        d = np.tile(d, [1,k_dist])
    return t, d, dist


def broadcast_long(x:np.ndarray, k:int) -> np.ndarray:
    """If we have an (n,1) array, convert to an (n,k) with duplicate columns"""
    if hasattr(x, 'shape'):
        if len(x.shape) == 1:
            return np.tile(t_long(x),[1,k])
        else:
            if x.shape[1] == 1:
                return np.tile(x,[1,k])
            else:
                return x
    else:
        return np.tile(t_long(x),[1,k])


def check_dist_str(dist:str or list) -> None:
    """CHECK THAT STRING BELONGS TO VALID DISTRIBUTION"""
    dist = str2lst(dist)
    assert all([d in dist_valid for d in dist]), f'dist must be one of: {", ".join(dist_valid)}'


def dist2vec(dist:str or list, scale:np.ndarray) -> list:
    """Ensure that a string or input array is a list of length k"""
    dist = str2lst(dist)
    check_dist_str(dist)
    k = max(len(dist), t_wide(scale).shape[1])
    if len(dist) == 1:
        dist = dist * k
    else:
        assert len(dist) == k, 'length of dist needs to align with scale'
    return dist


def is_vector(x:np.ndarray) -> None:
    """CHECKS THAT ARRAY HAS AT MOST POSSIBLE DIMENSION"""
    n_shape = len(x.shape)
    if n_shape <= 1:  # Scale or vector
        check = True
    elif n_shape == 2:
        if x.shape[1] == 1:
            check = False  # Is (k,1)
        else:
            check = False  # Is (p,k), k>1
    else:  # Must have 3 or more dimensions
        check = False
    assert check, 'Dimensionality not as expected'


def _get_p_k(t:np.ndarray) -> tuple[int, int]:
    """
    RETURN THE DIMENSIONALITY OF THE DATA INPUT ARRAY

    *NOTE, WHEN WE MOVE TO COVARIATES, INPUT WILL NEED TO CHANGE TO X
    """
    n_shape = len(t.shape)
    assert n_shape <= 2, 'Time-to-event can have at most 2-dimensions'
    if n_shape <= 1:
        k, p = 1, 2
    else:
        k, p = t.shape[1], 2
    return p, k


def shape_scale_2vec(shape_scale:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    SPLIT THE [p,k] matrix into a [1,k] and [p-1,k] row vector/matrix
    """
    shape, scale = t_wide(shape_scale[0]), t_wide(shape_scale[1:])
    return shape, scale


def format_t_d(t:np.ndarray, d:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    ENSURES THAT TIME/CENSORING ARE IN LONG FORM, AND SCALE/SHAPE ARE IN WIDE FORM
    """
    t_vec, d_vec = t_long(t), t_long(d)
    assert t_vec.shape == d_vec.shape, 'time and censoring matrix should be the same size'
    return t_vec, d_vec


def format_t_d_scale_shape(t:np.ndarray, d:np.ndarray, scale:np.ndarray, shape:np.ndarray or None)  -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ENSURES THAT TIME/CENSORING ARE IN LONG FORM, AND SCALE/SHAPE ARE IN WIDE FORM
    """
    t_vec, d_vec = format_t_d(t, d)
    scale, shape = t_wide(scale), t_wide(shape)
    k = scale.shape[1]
    t_vec = broadcast_long(t_vec, k)
    d_vec = broadcast_long(d_vec, k)
    return t_vec, d_vec, scale, shape


def t_wide(x:np.ndarray or float or pd.Series or None) -> np.ndarray:
    """CONVERT 1D ARRAY OR FLOAT TO A 1xK VECTOR"""
    if x is None:
        return x
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    n_shape = len(x.shape)
    if n_shape == 0:
        x = x.reshape([1, 1])
    if n_shape == 1:
        x = x.reshape([1, max(x.shape)])
    return x


def t_long(x:np.ndarray or float or pd.Series or None) -> np.ndarray:
    """CONVERT 1D ARRAY OR FLOAT TO A Kx1 VECTOR"""
    if x is None:
        return x
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    n_shape = len(x.shape)
    if n_shape == 0:
        x = x.reshape([1, 1])
    if n_shape == 1:
        x = x.reshape([max(x.shape), 1])
    return x


def flatten(x:np.ndarray or float or pd.Series) -> np.ndarray:
    """ Returns a flat array"""
    return np.array(x).flatten()


def len_of_none(x:np.ndarray or None) -> int:
    """Calculate length of array, or return 0 for nonetype"""
    l = 0
    if x is not None:
        l = len(x)
    return l


def param2array(x:float or np.ndarray or pd.Series) -> np.ndarray:
    """
    Checks that the input parameters to a distribution are either a float or a coercible np.ndarray
    """
    check = True
    # Check for conceivable floats
    lst_float = [float, np.float32, np.float64, int, np.int32, np.int64]
    if type(x) in lst_float:
        x = np.array([x])
    elif isinstance(x, list):
        x = np.array(x, dtype=float)
    elif isinstance(x, pd.Series):
        x = np.array(x, dtype=float)
    elif isinstance(x, np.ndarray):
        if len(x.shape) == 2:
            x = x.flatten()
    elif x is None:
        x = None
    else:
        check = False
    assert check, 'Input is not a float or coerible'
    return x

def coerce_to_Series(x) -> pd.Series:
    """
    Try to coerce an object x to a pd.Series. Currently supported for strings, integers, floats, lists, numpy arrays, and None's. 
    """
    if not isinstance(x, pd.Series):
        if isinstance(x, str) or isinstance(x, float) or isinstance(x, int):
            x = [x]
        elif isinstance(x, np.ndarray) or isinstance(x, list):
            if len(x) == 0:
                x = pd.Series(x, dtype=object)
            else:
                x = pd.Series(x)
        else:
            if len(x) > 0:  # Possible an index or some other pandas array
                x = pd.Series(x)
            elif x == None:
                x = pd.Series([], dtype=object)
            else:
                assert False, 'How did we get here??'
    return x


def check_type(x, tt: type, name: str=None) -> None:
    """
    Function checks whether object is of type tt, with variable named name

    Input
    -----
    x:         Any object
    tt:        Valid type for isinstance
    name:      Name of x (optional)
    """
    if name is None:
        name = 'x'
    assert isinstance(x, tt), f'{name} was not found to be: {tt}'


def vstack_pd(x: pd.DataFrame, y: pd.DataFrame, drop_idx=True) -> pd.DataFrame:
    """Quick wrapper to vertically stack two dataframes"""
    z = pd.concat(objs=[x, y], axis=0)
    if drop_idx:
        z.reset_index(drop=True, inplace=True)
    return z


def hstack_pd(x: pd.DataFrame, y: pd.DataFrame, drop_x:bool=True) -> pd.DataFrame:
    """
    Function allows for horizontal concatenation of two dataframes. If x and y share columns, then the columns from y will be favoured
    """
    check_type(x, pd.DataFrame, 'x')
    check_type(y, pd.DataFrame, 'y')
    cn_drop = list(intersect(x.columns, y.columns))
    if len(cn_drop) > 0:
        if drop_x:
            x = x.drop(columns=cn_drop)
        else:
            y = y.drop(columns=cn_drop)
    z = pd.concat(objs=[x, y], axis=1)
    return z



def setdiff(x: pd.Series, y: pd.Series) -> pd.Series:
    """R-like equivalent using pandas instead of numpy"""
    x = coerce_to_Series(x)
    y = coerce_to_Series(y)
    z = x[~x.isin(y)].reset_index(drop=True)
    return z

def intersect(x: pd.Series, y: pd.Series) -> pd.Series:
    """R-like equivalent using pandas instead of numpy"""
    x = coerce_to_Series(x)
    y = coerce_to_Series(y)
    z = x[x.isin(y)].reset_index(drop=True)
    return z


def str_subset(x:pd.Series, pat: str, regex:bool=True, case:bool=True, na:bool=False, neg:bool=False) -> pd.Series:
    """
    R-like regex string subset
    
    Input
    -----
    x:          Some array that can be converted to a pd.Series
    pat:        Some (regular expression) pattern to detect in x
    regex:      Should pattern be treated as regular expresson?
    case:       Should pattern be case-sensitive?
    na:         How should missing values be treated as matches?
    neg:        Should we the match be reversed?

    Returns
    ------
    A subset of x that matches pat
    """
    x = coerce_to_Series(x)
    if not x.dtype == object:
        x = x.astype(str)
    idx = x.str.contains(pat, regex=regex, case=case, na=na)
    if neg:
        z = x[~idx]
    else:   
        z = x[idx]
    z.reset_index(drop=True, inplace=True)
    return z