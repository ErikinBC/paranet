
def compare_lam_auroc(scale:float, censoring:float, T_dist_target:np.ndarray, n_censor_sim:int, seed:int or None=None) -> float:
    """
    For a given scale parameter, compare the probability that a given exponential distribution is larger than a comparison

    Inputs
    ------
    scale:              Scale parameter to test for exponential
    censoring:          The targeted censoring rate
    T_dist_target:      Sample from the distribution we want to censor
    n_censor_sim:       Number of samples to draw from for exponential censoring distribution


    Returns
    -------
    The MSE between the empirical AUROC and 1-target AUROC
    """
    T_dist_cens = rvs_T(n_sim=n_censor_sim, k=1, scale=scale, shape=None, dist='exponential', seed=seed)
    auroc = fast_auroc(T_dist_cens, T_dist_target)
    err = auroc - (1-censoring)
    err2 = err**2
    return err2


