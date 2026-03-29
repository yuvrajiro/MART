
# Moving Average Randomized Trees (MART)

![AAPLE](image.png)

A Python imblementation **Moving Average Randomized Trees (MART)**—an ensemble of  trees whose split criteria are driven by randomized exponential moving averages for financial time series forecasting.

## Features

- Integrates multiple EMA spans directly into tree‐split criteria  
- Preserves temporal dependencies without high‐dimensional lag features  
- Robust against whipsaw and fixed‐window limitations of classic MA crossover rules  
- Fast training and inference (Cython‐accelerated MACD computations)  
- Backtested on 19 diverse financial instruments  

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yuvrajiro/mart.git
   cd mart
    ```

2. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Compile the Cython MACD module**:

   ```bash
   python setup.py build_ext --inplace
   ```

4. Install the package:

   ```bash
   pip install -e .
   ```

## Usage

```python
from mart import MARTClassifier

# Load your price series P of length T
# Build and train the ensemble
model = EnsembleMART(n_estimators=100, max_depth=3, n_iter=200)
model.fit(price_series=P)

# Predict next‐step direction (+1 bullish, -1 bearish)
y_pred = model.predict(P)
```

### API

* `EnsembleMART(n_trees, max_depth, n_candidates, ...)`
  Constructor for the MART ensemble.
* `fit(price_series: np.ndarray)`
  Train on a 1D array of prices.
* `predict(price_series: np.ndarray)` -> `np.ndarray`
  Returns +1/–1 signals for each time step.

## Bash File

```bash
python proposed_exp.py  # Run MART
python experiments.py  # Run Baseline experiments
```

## Extra Resources (These files will not be included in the repository while publication of the code, but will be available on request, and included here for reviewers only)
- result_agg.ipynb : This File contains the code for aggregation of the reulsts, and create tables given in the paper.
- r_analysis.R : This file written in R, to do the statistical analysis of the results, because R has some of the best statistical packages available.
- build_tree_performance.py : This file contains the code computational efficiency analysis of the MART algorithm.





## License

This project is licensed under the MIT License.

## Contributing

Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a pull request on GitHub.
