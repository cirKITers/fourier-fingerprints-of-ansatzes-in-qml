"""Training loops as generators yielding per-step metrics and final results."""

from typing import Dict, Generator, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from qml_essentials.coefficients import Coefficients
from qml_essentials.model import Model
from torch.utils.data import DataLoader, TensorDataset


def _xp(a):
    return torch if isinstance(a, torch.Tensor) else jnp


def mse(prediction, target):
    """Mean squared error."""
    return ((prediction - target) ** 2).mean()


def null_loss(prediction, target):
    """Constant zero loss."""
    return 0.0


def kl_divergence(prediction, target):
    """KL divergence between normal distributions fitted to prediction and target."""
    var_p, var_t = prediction.var(), target.var()
    return 0.5 * (
        _xp(prediction).log(var_t / var_p)
        + (var_p + (prediction.mean() - target.mean()) ** 2) / var_t
        - 1
    )


def huber_loss(prediction, target, delta=1.0):
    """Huber loss with threshold `delta`."""
    xp = _xp(prediction)
    a = prediction - target
    return xp.where(
        xp.abs(a) <= delta, 0.5 * a**2, delta * (xp.abs(a) - 0.5 * delta)
    ).mean()


LOSSES = {f.__name__: f for f in (mse, null_loss, kl_divergence, huber_loss)}
VALID_METRICS = ("mse", "kl_divergence", "huber_loss")


def _cost(losses: Sequence[str], scalers: Sequence[float], prediction, target):
    # target and prediction are swapped as in the paper (matters for kl_divergence)
    return sum(s * LOSSES[f](target, prediction) for f, s in zip(losses, scalers))


def _track(best: Dict[str, float], metrics: Dict[str, float]) -> Dict[str, float]:
    return {k: min(v, best.get(k, np.inf)) for k, v in metrics.items()}


def train_qfm(
    model: Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    steps: int = 2000,
    learning_rate: float = 0.01,
    loss_function: Sequence[str] = ("mse", "null_loss"),
    loss_scaler: Sequence[float] = (1.0, 0.001),
) -> Generator[Dict[str, float], None, Tuple[jnp.ndarray, Dict[str, float]]]:
    """
    Full-batch training of a quantum Fourier model as in the paper.

    Uses Adam with the PennyLane defaults ($\\beta_2 = 0.99$) on the weighted sum
    of `loss_function`, starting from the current model parameters. The
    validation metrics are evaluated after each update.

    Parameters
    ----------
    model : Model
        The quantum Fourier model, receives the trained parameters.
    x_train, y_train : np.ndarray
        Training inputs of shape (N, D) and targets of shape (N,).
    x_valid, y_valid : np.ndarray
        Validation inputs and targets.
    steps : int, optional
        Number of optimization steps.
    learning_rate : float, optional
        Adam learning rate.
    loss_function : Sequence[str], optional
        Names of the losses in `LOSSES`.
    loss_scaler : Sequence[float], optional
        Scale of each loss.

    Yields
    ------
    Dict[str, float]
        step, cost (before the update) and {mse,kl_divergence,huber_loss}_valid.

    Returns
    -------
    Tuple[jnp.ndarray, Dict[str, float]]
        Trained parameters and the minimum of each validation metric over the
        steps as {metric}_valid_min.
    """

    def predict(params, x):
        return model.apply(params=params, inputs=x, force_mean=True).reshape(-1)

    def cost(params):
        return _cost(loss_function, loss_scaler, predict(params, x_train), y_train)

    def validate(params):
        prediction = predict(params, x_valid)
        return {f"{m}_valid": LOSSES[m](prediction, y_valid) for m in VALID_METRICS}

    step_fn = jax.jit(jax.value_and_grad(cost))
    validate = jax.jit(validate)
    opt = optax.adam(learning_rate, b2=0.99)
    params = model.params
    state = opt.init(params)
    best = {}
    for step in range(steps):
        cost_value, grads = step_fn(params)
        updates, state = opt.update(grads, state)
        params = optax.apply_updates(params, updates)
        metrics = {k: float(v) for k, v in validate(params).items()}
        best = _track(best, metrics)
        yield {"step": step, "cost": float(cost_value), **metrics}

    model.params = params
    return params, {f"{k}_min": v for k, v in best.items()}


def train_fourier_series(
    model: Model,
    x: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    steps: int = 3000,
    learning_rate: float = 1e-4,
    unnormalized_target: bool = False,
    mask: Optional[np.ndarray] = None,
) -> Generator[Dict[str, float], None, Tuple[jnp.ndarray, Dict[str, float]]]:
    """
    Full-batch MSE training of the encoding study.

    Uses Adam with the optax defaults, starting from the current model
    parameters. With `mask`, the gradients of the other parameters are zero,
    so Adam keeps them at their current values. train_fmse is the mean
    absolute difference between the model spectrum and the spectrum of the
    target, i.e. `coefficients` divided by their number.

    Parameters
    ----------
    model : Model
        The quantum Fourier model, receives the trained parameters.
    x, y : np.ndarray
        Inputs of shape (N, D) and targets of shape (N,).
    coefficients : np.ndarray
        Target coefficients, ordered like the shifted model spectrum.
    steps : int, optional
        Number of optimization steps.
    learning_rate : float, optional
        Adam learning rate.
    unnormalized_target : bool, optional
        Compare against the unnormalized `coefficients` as in the encoding
        study of the thesis.
    mask : Optional[np.ndarray], optional
        Boolean mask of the shape of the model parameters, True for the
        trained parameters; None trains all.

    Yields
    ------
    Dict[str, float]
        step, train_mse and train_fmse after the update.

    Returns
    -------
    Tuple[jnp.ndarray, Dict[str, float]]
        Trained parameters and the final train_mse and train_fmse.
    """

    def cost(params):
        return mse(model.apply(params=params, inputs=x, force_mean=True).reshape(-1), y)

    grad_fn = jax.jit(jax.grad(cost))
    cost = jax.jit(cost)
    target = coefficients if unnormalized_target else coefficients / coefficients.size
    opt = optax.adam(learning_rate)
    params = model.params
    state = opt.init(params)
    for step in range(steps):
        grads = grad_fn(params)
        if mask is not None:
            grads = grads * mask
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        model.params = params
        spectrum, _ = Coefficients.get_spectrum(model, shift=True, params=params)
        metrics = {
            "train_mse": float(cost(params)),
            "train_fmse": float(np.mean(np.abs(spectrum - target))),
        }
        yield {"step": step, **metrics}

    return params, metrics


class HEPRegressor(torch.nn.Module):
    """Multilayer perceptron with LeakyReLU activations and a scalar output."""

    def __init__(self, n_features: int, width: int = 8, depth: int = 2):
        super().__init__()
        layers = [torch.nn.Linear(n_features, width), torch.nn.LeakyReLU()]
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(width, width), torch.nn.LeakyReLU()]
        layers += [torch.nn.Linear(width, 1)]
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze()


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    seed: int = 1000,
    width: int = 8,
    depth: int = 2,
    steps: int = 150,
    learning_rate: float = 0.005,
    batch_size: int = 256,
    loss_function: Sequence[str] = ("kl_divergence", "huber_loss"),
    loss_scaler: Sequence[float] = (1.0, 0.001),
    broadcast_targets: bool = False,
) -> Generator[Dict[str, float], None, Tuple[HEPRegressor, Dict[str, float]]]:
    """
    Mini-batch training of the classical baseline as in the paper.

    Uses Adam with weight decay $10^{-5}$ and a plateau scheduler (factor 0.75,
    patience 6) stepped every 4th epoch on the mean batch cost.

    Parameters
    ----------
    x_train, y_train : np.ndarray
        Training inputs of shape (N, D) and targets of shape (N,).
    x_valid, y_valid : np.ndarray
        Validation inputs and targets.
    seed : int, optional
        Seed of the weight initialization and the batch shuffling.
    width : int, optional
        Neurons per hidden layer.
    depth : int, optional
        Number of hidden layers.
    steps : int, optional
        Number of epochs.
    learning_rate : float, optional
        Initial Adam learning rate.
    batch_size : int, optional
        Batch size.
    loss_function : Sequence[str], optional
        Names of the losses in `LOSSES`.
    loss_scaler : Sequence[float], optional
        Scale of each loss.
    broadcast_targets : bool, optional
        Keep a trailing target axis as in the paper, so the Huber loss in the
        cost compares all prediction-target pairs of a batch.

    Yields
    ------
    Dict[str, float]
        step, cost (mean over the batches) and {mse,kl_divergence,huber_loss}_valid.

    Returns
    -------
    Tuple[HEPRegressor, Dict[str, float]]
        Trained model and the minimum of each validation metric over the epochs
        as {metric}_valid_min.
    """
    torch.manual_seed(seed)
    model = HEPRegressor(x_train.shape[1], width=width, depth=depth)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.75, patience=6)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(y_train[:, None] if broadcast_targets else y_train),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    best = {}
    for step in range(steps):
        cost_value = 0.0
        for x, y in loader:
            cost = _cost(loss_function, loss_scaler, model(x), y)
            opt.zero_grad()
            cost.backward()
            opt.step()
            cost_value += cost.item()
        cost_value /= len(loader)
        if step % 4 == 0:
            sched.step(cost_value)

        with torch.no_grad():
            prediction = model(torch.from_numpy(x_valid)).numpy()
        metrics = {
            f"{m}_valid": float(LOSSES[m](prediction, y_valid)) for m in VALID_METRICS
        }
        best = _track(best, metrics)
        yield {"step": step, "cost": cost_value, **metrics}

    return model, {f"{k}_min": v for k, v in best.items()}
