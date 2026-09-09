# Multi-Location Regression

Simulation code for studying a multi-location attention model and its high-dimensional limit.

## Model

The data matrix is $X \in \mathbb{R}^{L \times d}$, with target

$$
y = \frac{1}{d}(X_{\epsilon_1^*})^T V^* X_{\epsilon_2^*}.
$$

The input distribution is tilted according to

$$
P(X\mid\epsilon_1^*,\epsilon_2^*) \propto
g(\epsilon_1^*,\epsilon_2^*,\chi^*)P_0(X),
\qquad
g=\exp\left(\gamma\chi^*_{\epsilon_1^*,\epsilon_2^*}\right),
$$

where $\chi^* = XW^*X^T/\sqrt{d}$. The model prediction is

$$
\hat y=\sigma(XW_1X^T)_1^T XVX^T\sigma(XW_2X^T)_1,
$$

with softmax activation by default. All matrices are rank one:
$M=u_Mu_M^T$ for $M\in\{W^*,V^*,W_1,V,W_2\}$.

The main parameters are $L=3$, $d=10^4$, $\gamma=0.49$, and
$\epsilon_1^*=0$, $\epsilon_2^*=1$.

## Files

- `Order_parameters.py`: optimizes the asymptotic loss by Monte Carlo integration and L-BFGS. It saves `L_block_opt.npy` and `Q_opt.npy`, where $Q_{MN}=u_M^Tu_N/d$.
- `sgd_rank1.py`: trains the finite-dimensional rank-one model with online SGD and fresh tilted samples at every epoch.
- `environment.yaml`: Conda environment specification.

## Installation

```powershell
conda env create -f environment.yaml
conda activate MLR_conda_env
```

PyTorch uses CUDA automatically when available. The default experiments are computationally intensive and may require a GPU with sufficient memory.

## Running the simulations

Asymptotic order-parameter optimization:

```powershell
python Order_parameters.py
```

Finite-dimensional online SGD:

```powershell
python sgd_rank1.py
```

The SGD script uses $d=10^4$, batch size $N=10^4$, $10^6$ epochs, and learning rate $0.1$. It saves the trained model, training losses, and validation losses. Both scripts use PyTorch seed `0`.

For a quick test, reduce `nMc` and `lbfgs_steps` in `Order_parameters.py`, or set smaller values for `D`, `N`, and `EPOCHS` in `sgd_rank1.py`. These reduced settings do not reproduce the poster results.

The saved losses and covariance matrices can be used to reproduce the convergence and order-parameter comparisons shown in the poster.