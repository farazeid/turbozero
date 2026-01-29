from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import optax
from einops import rearrange
from flax.training.train_state import TrainState

from core.memory.replay_memory import BaseExperience


def az_default_loss_fn(
    params: chex.ArrayTree,
    train_state: TrainState,
    experience: BaseExperience,
    l2_reg_lambda: float = 0.0001,
) -> Tuple[chex.Array, Tuple[chex.ArrayTree, optax.OptState]]:
    """Implements the default AlphaZero loss function.

    = Policy Loss + Value Loss + L2 Regularization
    Policy Loss: Cross-entropy loss between predicted policy and target policy
    Value Loss: L2 loss between predicted value and target value

    Args:
    - `params`: the parameters of the neural network
    - `train_state`: flax TrainState (holds optimizer and other state)
    - `experience`: experience sampled from replay buffer
        - stores the observation, target policy, target value
    - `l2_reg_lambda`: L2 regularization weight (default = 1e-4)

    Returns:
    - (loss, (aux_metrics, updates))
        - `loss`: total loss
        - `aux_metrics`: auxiliary metrics (policy_loss, value_loss)
        - `updates`: optimizer updates
    """

    # get batch_stats if using batch_norm
    variables = (
        {"params": params, "batch_stats": train_state.batch_stats}
        if hasattr(train_state, "batch_stats")
        else {"params": params}
    )
    mutables = ["batch_stats"] if hasattr(train_state, "batch_stats") else []

    # get predictions
    (pred_policy, pred_value), updates = train_state.apply_fn(
        variables, x=experience.observation_nn, train=True, mutable=mutables
    )

    # set invalid actions in policy to -inf
    pred_policy = jnp.where(
        experience.policy_mask, pred_policy, jnp.finfo(jnp.float32).min
    )

    # compute policy loss
    policy_loss = optax.softmax_cross_entropy(
        pred_policy, experience.policy_weights
    ).mean()
    # select appropriate value from experience.reward
    current_player = experience.cur_player_id
    target_value = experience.reward[
        jnp.arange(experience.reward.shape[0]), current_player
    ]
    # compute MSE value loss
    value_loss = optax.l2_loss(pred_value.squeeze(), target_value).mean()

    l2_reg = l2_reg_lambda * jax.tree_util.tree_reduce(
        lambda x, y: x + y, jax.tree_map(lambda x: (x**2).sum(), params)
    )

    # total loss
    loss = policy_loss + value_loss + l2_reg
    aux_metrics = {"policy_loss": policy_loss, "value_loss": value_loss}
    return loss, (aux_metrics, updates)


def katago_loss_fn(
    params: chex.ArrayTree,
    train_state: TrainState,
    batch: Dict[str, chex.Array],
    l2_reg_lambda: float = 0.0001,
) -> Tuple[chex.Array, Tuple[Dict[str, chex.Array], Dict[str, Any]]]:
    """Implements the KataGo multi-head loss function.

    Args:
    - `params`: the parameters of the neural network
    - `train_state`: flax TrainState
    - `batch`: dictionary containing inputs and targets
        - 'binaryInputNCHW', 'globalInputNC', etc.
    - `l2_reg_lambda`: L2 regularization weight

    Returns:
    - (loss, (aux_metrics, updates))
    """
    variables = (
        {"params": params, "batch_stats": train_state.batch_stats}
        if hasattr(train_state, "batch_stats")
        else {"params": params}
    )
    mutables = ["batch_stats"] if hasattr(train_state, "batch_stats") else []

    # get predictions
    # KataGoNetwork returns (policy_logits, value, ownership, score)
    (pred_policy, pred_value, pred_ownership, pred_score), updates = (
        train_state.apply_fn(
            variables, x=batch["binaryInputNCHW"], train=True, mutable=mutables
        )
    )

    # 1. Policy Loss
    # policyTargetsNCMove is (N, 2, 362) or (N, 362)
    labels = batch["policyTargetsNCMove"]
    if len(labels.shape) == 3:
        # Use second channel (MCTS distribution)
        labels = rearrange(labels, "b c p -> c b p")[1]

    policy_loss = optax.softmax_cross_entropy(pred_policy, labels).mean()

    # 2. Value Loss
    # pred_value: (N, 3) = [win, loss, draw] or (N, 1)
    # globalTargetsNC: C0 is win estimate
    target_value = batch["globalTargetsNC"][:, :3]  # win/loss/draw targets
    if pred_value.shape[-1] == 3:
        # Full 3-output: compare win/loss/draw
        value_loss = optax.l2_loss(pred_value, target_value).mean()
    else:
        # Single output: compare to win estimate
        value_loss = optax.l2_loss(pred_value[:, 0], target_value[:, 0]).mean()

    # 3. Ownership Loss
    # pred_ownership: (N, 19, 19, 1)
    # target_ownership: (N, 19, 19, 5) - C0 is ownership
    target_ownership = batch["valueTargetsNCHW"]
    if target_ownership.shape[-1] > 1:
        target_ownership = target_ownership[:, :, :, 0:1]

    ownership_loss = optax.l2_loss(pred_ownership, target_ownership).mean()

    # 4. Score Loss
    # pred_score: (N, 6) misc outputs or (N, 1)
    # globalTargetsNC: C58 is raw scoremean from neural net
    target_score = batch["globalTargetsNC"][:, 58:64]  # 6 misc values
    if pred_score.shape[-1] == 6:
        score_loss = optax.l2_loss(pred_score, target_score).mean()
    else:
        # Single output
        score_loss = optax.l2_loss(pred_score[:, 0], target_score[:, 0]).mean()

    # 5. L2 Regularization
    l2_reg = l2_reg_lambda * jax.tree_util.tree_reduce(
        lambda x, y: x + y, jax.tree_map(lambda x: (x**2).sum(), params)
    )

    # Total Loss (weighted)
    loss = policy_loss + value_loss + ownership_loss + score_loss + l2_reg

    aux_metrics = {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "ownership_loss": ownership_loss,
        "score_loss": score_loss,
    }

    return loss, (aux_metrics, updates)


def shapley_loss_fn(
    params: chex.ArrayTree,
    train_state: TrainState,
    batch: Dict[str, chex.Array],
    importance_weights: chex.Array | None = None,
    l2_reg_lambda: float = 0.0001,
) -> Tuple[chex.Array, Tuple[Dict[str, chex.Array], Dict[str, Any]]]:
    """Implements the FastSVERL Shapley Model loss function (Equation 10/19).

    This loss trains exactly one Shapley model (Behaviour, Outcome, or Prediction)
    by comparing its masked spatial sum to the target characteristic values.

    Args:
    - `params`: parameters of the Shapley model
    - `train_state`: flax TrainState
    - `batch`: dictionary containing:
        - 'observation': (N, H, W, C) input board
        - 'coalition_mask': (N, H, W, 1) binary mask of known features
        - 'target_char_vals': (N, num_outputs) characteristic values for current coalition
        - 'null_char_vals': (N, num_outputs) characteristic values for empty coalition
    - `importance_weights`: Optional (N,) array of normalised importance sampling weights
        for off-policy training. If provided, uses weighted MSE loss.
    - `l2_reg_lambda`: L2 regularization weight

    Returns:
    - (loss, (aux_metrics, updates))
    """
    variables = (
        {"params": params, "batch_stats": train_state.batch_stats}
        if hasattr(train_state, "batch_stats")
        else {"params": params}
    )
    mutables = ["batch_stats"] if hasattr(train_state, "batch_stats") else []

    # Get model output (phi): (N, H, W, num_outputs)
    phi, updates = train_state.apply_fn(
        variables,
        x=batch["observation"],
        global_input=batch.get("global_input"),
        mask=batch["coalition_mask"],
        train=True,
        mutable=mutables,
    )

    # 1. Prediction for each output channel (action/scalar)
    # Sum spatial attributions over the coalition mask
    # predictions: (N, num_outputs)
    predictions = jnp.sum(batch["coalition_mask"] * phi, axis=(1, 2))

    # 2. Target for each output channel
    # targets: (N, num_outputs)
    targets = batch["target_char_vals"] - batch["null_char_vals"]

    # 3. Compute loss (weighted or unweighted MSE)
    # Average over outputs first
    errors = jnp.mean(jnp.square(predictions - targets), axis=-1)  # (N,)

    if importance_weights is not None:
        # Weighted MSE: sum(w_i * err_i)
        # Note: importance_weights are expected to be self-normalised (sum to 1)
        shapley_loss = jnp.sum(importance_weights * errors)
    else:
        # Standard unweighted MSE
        shapley_loss = jnp.mean(errors)

    # 4. L2 Regularization
    l2_reg = l2_reg_lambda * jax.tree_util.tree_reduce(
        lambda x, y: x + y, jax.tree_map(lambda x: (x**2).sum(), params)
    )

    loss = shapley_loss + l2_reg

    aux_metrics = {
        "loss": loss,
        "shapley_loss": shapley_loss,
        "l2_reg": l2_reg,
    }

    return loss, (aux_metrics, updates)
