from numba import njit
import numpy as np


@njit(cache=True)
def cagr_gain_jit(
    series: np.ndarray,
    w: np.ndarray,
    arr0: np.ndarray,
    arr1: np.ndarray,
    arrm1: np.ndarray,
    sign: int,
    periods_per_year: int = 252
) -> float:
    """
    Compute a single split CAGR metric for this node, relative to its parent sign.

    We calculate:
      - parent return factor over all signals at this node,
      - split return factor over the same signals but without parent sign,
      - annualize both with the count of signals,
      - return (cagr_split - cagr_parent).

    Parameters
    ----------
    series : 1D array of prices
    w      : 1D array of weights
    arr0   : indices for neutral class (ignored)
    arr1   : indices for buy signals
    arrm1  : indices for sell signals
    sign   : +1 for parent long, -1 for parent short
    periods_per_year : periods per year (e.g. 252)
    """
    n = series.shape[0]
    # assert arr1 and arrm1 are disjoint

    # number of trading events in this node
    n_signals_parent = arr1.shape[0] + arrm1.shape[0]
    n_signals_split = arr1.shape[0] + arrm1.shape[0]
    if n_signals_split == 0:
        return 0.0

    # 1) Parent return factor (applies parent sign to each signal)
    parent_factor = 1.0
    split_factor = 1.0
    # buys
    for k in range(arr1.shape[0]):
        idx = arr1[k]
        if 0 <= idx < n-1:
            r = series[idx+1]/series[idx] - 1.0
            parent_factor *= 1.0 + sign * w[idx] * r
            split_factor *= 1.0 + w[idx] * r
    # sells
    for k in range(arrm1.shape[0]):
        idx = arrm1[k]
        if 0 <= idx < n-1:
            r = series[idx+1]/series[idx] - 1.0
            parent_factor *= 1.0 + sign * w[idx] * r
            split_factor *= 1.0 - w[idx] * r

    exponent_parent = periods_per_year / float(n_signals_parent)
    exponent_split  = periods_per_year / float(n_signals_split)
    cagr_parent = parent_factor**exponent_parent - 1.0
    cagr_split  = split_factor**exponent_split - 1.0

    # 4) return difference
    return cagr_split - cagr_parent


