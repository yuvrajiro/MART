class DecisionNode:
    """
    Internal node: stores split parameters and children.
    """
    def __init__(self, fast_span, slow_span, signal_span, hist_quantile,div_lookback, left_child, right_child , sign=None, conf=None):
        self.fast_span = fast_span
        self.slow_span = slow_span
        self.signal_span = signal_span
        self.hist_quantile = hist_quantile
        self.div_lookback = div_lookback
        self.left = left_child
        self.right = right_child
        self.conf = conf  # probability of sign

class LeafNode:
    """
    Leaf node: holds the indices that fall here.
    """
    def __init__(self, indices, sign, conf):
        self.indices = indices
        self.sign = sign  # 1 for buy, -1 for sell
        self.conf = conf  # probability of sign