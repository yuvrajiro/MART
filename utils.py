import numpy as np
import os
import sys
import pandas as pd
import yfinance
from tree_struct import DecisionNode, LeafNode



def get_x_y(series, test_index, lookback=5):
    """
    
    Build train/test feature matrices (X) and label vectors (y) from a 1D price series.
    
    Features (X) are sliding windows of the last `lookback` prices.
    Labels (y) are +1 if next-bar return > 0, else -1 (flat/down = -1).
    
    Args:
        series      (array-like):  1D array of prices, length N
        test_index  (array-like):  list or array of original-series indices to use for testing
        lookback      (int):       number of past bars to include in each feature vector
        
    Returns:
        X_train (ndarray): shape (n_train, lookback)
        y_train (ndarray): shape (n_train,)
        X_test  (ndarray): shape (n_test,  lookback)
        y_test  (ndarray): shape (n_test,)
    """
    prices = np.asarray(series, dtype=float)
    N = len(prices)
    
    # 1) Compute the next-bar direction: diff[i] = prices[i+1] - prices[i]
    diffs = np.diff(prices)                          # length = N-1
    signal = np.sign(diffs)                          # -1, 0, +1
    signal[signal == 0] = +1                         # treat flat as -1
    
    # 2) Build sliding windows of length `lookback`
    #    window 0 covers prices[0:lookback], window 1 covers prices[1:lookback+1], etc.
    X = np.array([prices[i : i + lookback] 
                  for i in range(N - lookback)])     # shape = (N - lookback, lookback)
    
    # 3) Align labels so that y[i] corresponds to the movement out of the last price in X[i]
    #    That movement is diffs[i + lookback - 1].
    y = signal[lookback - 1 :]                       # length = N - lookback
    
    # 4) Find how many windows precede the first test index
    first_test = int(test_index[0])
    split = first_test - lookback
    if split < 0:
        raise ValueError(f"lookback {lookback} is too large for test_index start {first_test}")
    
    # 5) Slice into train vs. test
    X_train, y_train = X[:split],    y[:split]
    X_test,  y_test  = X[split:],    y[split:]
    
    # 6) Sanity checks
    assert len(y_train) == len(X_train)
    assert len(y_test)  == len(X_test)
    assert len(y_test)  == len(test_index), (
        f"expected {len(test_index)} test labels, got {len(y_test)}"
    )
    
    return X_train, y_train, X_test, y_test


def streak_weight(arr):
    """
    This function defines a streak as a run of consecutive increments or decrements.
    It returns an array of the same length as the input, where each element is the
    number of consecutive equal‐sign differences starting at that position.
    The last element always gets a weight of 1.

    Example:
        >>> streak_weight([1, 2, 3, 2, 1])
        [2, 1, 2, 1, 1]
    """
    n = len(arr)
    if n == 0:
        return []

    weights = []
    for i in range(n - 1):
        diff = arr[i+1] - arr[i]
        # no increment/decrement → no streak beyond itself
        if diff == 0:
            weights.append(1)
            continue

        # determine streak sign (+1 or -1)
        sign = 1 if diff > 0 else -1
        count = 1  # we've already got one diff of this sign

        # extend forward as long as diffs have the same sign
        for j in range(i+1, n - 1):
            next_diff = arr[j+1] - arr[j]
            next_sign = 1 if next_diff > 0 else -1 if next_diff < 0 else 0
            if next_sign == sign:
                count += 1
            else:
                break

        weights.append(count)

    # last element: no forward neighbor → streak weight of 1
    weights.append(1)
    return weights

def get_random_candidates(series_len, n_iter, rng):
    """
    Generate random (fast, slow, signal) span triples for MACD.

    - fast_span < slow_span
    - signal_span < slow_span
    - all spans ≥ 1 and ≤ series_len

    Parameters
    ----------
    series_len : int
        Maximum look-back (e.g. length of the series).
    n_iter : int
        Number of random candidates to return.
    rng : numpy.random.Generator
        A random number generator for reproducibility.

    Returns
    -------
    List of tuples (fast_span, slow_span, signal_span)
    """
    # Build the full universe of valid (fast, slow) pairs
    pairs = [(f, s) for f in range(2, series_len)
                   for s in range(f+1, series_len+1)]
    # Randomly choose up to n_iter of those pairs
    chosen_pairs = rng.choice(len(pairs), size=min(n_iter, len(pairs)), replace=False)
    
    candidates = []
    for idx in chosen_pairs:
        f, s = pairs[idx]
        # For each (fast, slow), pick a signal span in [1, slow-1]
        
        sig = rng.integers(1, f)  # exclusive upper bound

        candidates.append((f, s, int(sig)))
    
    return candidates



def deep_getsizeof(obj, seen=None):
    """Recursively find the total memory footprint of an object and all of its contents."""
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = sys.getsizeof(obj)

    # For built-in containers, iterate their contents
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += deep_getsizeof(k, seen)
            size += deep_getsizeof(v, seen)
    elif hasattr(obj, '__dict__'):
        size += deep_getsizeof(vars(obj), seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        for item in obj:
            size += deep_getsizeof(item, seen)

    return size


def plot_tree(node, graph=None, parent_id=None, node_id=0):
    """
    Recursively plots the decision tree.

    Args:
        node: DecisionNode or LeafNode instance.
        graph: Graphviz Digraph object.
        parent_id: ID of the parent node.
        node_id: ID for the current node (unique integer).
    Returns:
        The updated graph and next node_id.
    """
    if graph is None:
        graph = Digraph(format='png')
    
    current_id = str(node_id)

    if isinstance(node, DecisionNode):
        label = f"Decision\n Fast: {node.fast_span}\n" \
                f" Slow: {node.slow_span}\n Signal: {node.signal_span}\n"
        graph.node(current_id, label, shape='box')
        
        # Connect to parent if exists
        if parent_id is not None:
            graph.edge(parent_id, current_id)

        # Recurse left and right
        graph, next_id = plot_tree(node.left, graph, current_id, node_id + 1)
        graph, next_id = plot_tree(node.right, graph, current_id, next_id)
        
    elif isinstance(node, LeafNode):
        label = f"Leaf\nSign: {'Buy' if node.sign == 1 else 'Sell'}\nProb: {node.conf:.2f}\nSamples: {len(node.indices)}"
        graph.node(current_id, label, shape='ellipse', style='filled', fillcolor='lightgrey')

        if parent_id is not None:
            graph.edge(parent_id, current_id)
        
        next_id = node_id + 1

    else:
        raise TypeError("Unknown node type!")

    return graph, next_id

