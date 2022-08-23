"""
UTILITY FUNCTIONS
"""

# External modules
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata

# List of currently supported distributions
dist_valid = ['exponential', 'weibull', 'gompertz']


def fast_auroc(x:np.ndarray, y:np.ndarray) -> float:
    n_x, n_y = len(x), len(y)
    x, y = flatten(x), flatten(y)
    df = pd.DataFrame({'grp':np.repeat([1,0],[n_x, n_y]), 's':np.concatenate((x,y))})
    df['r'] = rankdata(df['s'])
    auroc = (df.query('grp==1')['r'].sum() - n_x*(n_x+1)/2) / (n_x*n_y)
    return auroc


def gg_save(fn:str, fold:str, gg, width:float=5, height:float=4):
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


def check_interval(x:np.ndarray or float or pd.Series, low:int or float, high:int or float) -> None:
    """Check: low <= x <= high"""
    assert np.all((x >= low) & (x <= high)), 'x is not between [low,high]'


def check_dist_str(dist:str) -> None:
    """CHECK THAT STRING BELONGS TO VALID DISTRIBUTION"""
    assert isinstance(dist, str)
    assert dist in dist_valid, f'dist must be one of: {", ".join(dist_valid)}'


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


def get_p_k(t:np.ndarray) -> tuple[int, int]:
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


def format_t_d(t:np.ndarray, d:np.ndarray, dist:str) -> tuple[np.ndarray, np.ndarray]:
    """
    ENSURES THAT TIME/CENSORING ARE IN LONG FORM, AND SCALE/SHAPE ARE IN WIDE FORM
    """
    check_dist_str(dist)
    t_vec, d_vec = t_long(t), t_long(d)
    assert t_vec.shape == d_vec.shape, 'time and censoring matrix should be teh same size'
    return t_vec, d_vec


def format_t_d_scale_shape(t:np.ndarray, d:np.ndarray, scale:np.ndarray, shape:np.ndarray or None, dist:str)  -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ENSURES THAT TIME/CENSORING ARE IN LONG FORM, AND SCALE/SHAPE ARE IN WIDE FORM
    """
    t_vec, d_vec = format_t_d(t, d, dist)
    scale, shape = t_wide(scale), t_wide(shape)
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

def coerce_to_Series(x):
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


def check_type(x, tt: type, name: str=None):
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



def setdiff(x: pd.Series, y: pd.Series):
    """R-like equivalent using pandas instead of numpy"""
    x = coerce_to_Series(x)
    y = coerce_to_Series(y)
    z = x[~x.isin(y)].reset_index(drop=True)
    return z

def intersect(x: pd.Series, y: pd.Series):
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