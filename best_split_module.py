import numpy as np
from information_gain import cagr_gain_jit
from utils import get_random_candidates
from macd import compare_macd

def best_split(
        series: np.ndarray,
        indices: np.ndarray,
        sign : int,
        w: np.ndarray = None,
        random_state: int = 42,
        n_iter: int = 500,
        na_pass: str = 'both',
        hist_quantile: float = 0,
        div_lookback: int = 0,
        ):
    """
    Find best moving-average split by maximizing information gain.
    Returns tuple of best parameters and split metadata.
    """
    if w is None:
        w = np.ones(len(series), dtype=np.float64)
        w = w / np.sum(w)  # Normalize weights

    series_len = len(series)
    best_gain = 0
    best_params = (None, None, None)
    best_split_data = ([1], [1])
    rng = np.random.default_rng(random_state)


    candidates = get_random_candidates(series_len, n_iter, rng)
    for fast_span,slow_span,signal_span in candidates:
        # print(f"Checking for candidates: Short: {fast_span}, Long: {slow_span}, Signal: {signal_span}, Hist Quantile: {hist_quantile}, Div Lookback: {div_lookback}")
        nan_idx, buy_idx, sell_idx = compare_macd(
                series,
                fast_span=fast_span,
                slow_span=slow_span,
                signal_span=signal_span,
                hist_quantile=hist_quantile,
                div_lookback=div_lookback,
                indices=indices)
        ig = cagr_gain_jit(series,
                w,  
                nan_idx,
                buy_idx,
                sell_idx,
                sign)

        if ig > best_gain:
            #print(f"New best found: Short: {fast_span}, Long: {slow_span}, Signal: {signal_span} => CAGR Gain: {ig:.6%}")
            best_gain = ig
            best_params = (fast_span, slow_span, signal_span)
            left_idx = np.concatenate([buy_idx, nan_idx]) if na_pass in ('left','both') else buy_idx
            right_idx = np.concatenate([sell_idx, nan_idx]) if na_pass in ('right','both') else sell_idx
            best_split_data = (left_idx, right_idx)



    fast_span, slow_span, signal_span = best_params
    left_idx, right_idx = best_split_data
    

    return fast_span, slow_span, signal_span, left_idx, right_idx, best_gain