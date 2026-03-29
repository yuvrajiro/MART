import numpy as np
from best_split_module import best_split
from macd import compare_macd
from tree_struct import DecisionNode, LeafNode



def build_tree(
    series, w, indices,sign,
    min_samples_split=2,
    min_samples_leaf=1,
    max_depth=None,
    depth=0,
    random_state=42,
    n_iter=500,
    na_pass='both',
    hist_quantile=0,
    div_lookback=0,
) -> DecisionNode | LeafNode:
    """
    Recursively build a binary tree by choosing the best split at each node.
    Returns a DecisionNode or LeafNode.
    """
    current_series = series[indices]
    prob = (np.where(current_series[1:] > current_series[:-1], 1, -1) == sign).mean()
    sign_leaf = sign

    # stop if too few splittable samples OR we've hit max_depth
    if len(indices) < min_samples_split or (max_depth is not None and depth >= max_depth):
        return LeafNode(indices, sign_leaf, prob)

    
    fast_span, slow_span, signal_span, left_idx, right_idx, best_gain = best_split(
        series, indices, sign, w, random_state=random_state,
        n_iter=n_iter,na_pass=na_pass,hist_quantile=hist_quantile, div_lookback=div_lookback
    )
    if best_gain <= 0:
        return LeafNode(indices, sign_leaf, prob)
    if len(left_idx) < min_samples_leaf or len(right_idx) < min_samples_leaf:
        return LeafNode(indices, sign_leaf, prob)
    

    # Recursively build children
    left_child = build_tree(
        series, w, left_idx, 1,
        min_samples_split, min_samples_leaf,
        max_depth, depth+1,
        random_state, n_iter, na_pass, 
        hist_quantile=hist_quantile, div_lookback=div_lookback)
    right_child = build_tree(
        series, w, right_idx, -1,
        min_samples_split, min_samples_leaf,
        max_depth, depth+1,
        random_state, n_iter, na_pass, 
        hist_quantile=hist_quantile, div_lookback=div_lookback)
    return DecisionNode(fast_span, slow_span, signal_span,hist_quantile,div_lookback, left_child, right_child, conf=prob)




def predict_tree_one_iter(node, series):
    current = node
    last_idx = len(series) - 1
    while not isinstance(current, LeafNode):
        array_0_indices, array_1_indices, array_minus1_indices = compare_macd(
            series,
            fast_span=current.fast_span,
            slow_span=current.slow_span,
            signal_span=current.signal_span,
            hist_quantile=current.hist_quantile,
            div_lookback=current.div_lookback,
            indices = np.array([last_idx], dtype=np.intp)
        )
        if last_idx in array_1_indices:
            current = current.left
        elif last_idx in array_minus1_indices:
            current = current.right
        else:
            raise ValueError("Last index not found in any child indices")
    return current.sign, current.conf
    
def predict_tree(node, series, training_series):
    """
    Predict the sign for a given series using the decision tree.
    Returns 1 for buy, -1 for sell, or 0 for no prediction.
    The prediction is made iteratively, also the first prediction can be
    neglected or shifted since H_t is predicting y_t, the prediction
    y_t can be used for the next prediction H_{t+1} to predict y_{t+1}
    while calulating CAGR this must be taken care.
    """

    combined = np.concatenate([training_series, series])
    Ntrain = len(training_series)
    first_value = series[0]
    predict_sign = []
    predict_conf = []
    for i in range(len(series)):
        window = combined[: Ntrain + i]  
        if i == 0 :
            last_window_value = window[-1]
            assert first_value != last_window_value, f"Leakage {first_value} and {last_window_value}"
        sign, conf = predict_tree_one_iter(node, window)
        predict_sign.append(sign)
        predict_conf.append(conf)


    return np.array(predict_sign), np.array(predict_conf)
