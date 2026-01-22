from typing import Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from core.memory.replay_memory import BaseExperience
from src.shapley import (
    compute_behaviour_characteristic_loss,
    compute_behaviour_shapley_loss,
    compute_prediction_shapley_loss,
)


def az_default_loss_fn(
    params: chex.ArrayTree,
    train_state: TrainState,
    experience: BaseExperience,
    key: jax.Array,
    l2_reg_lambda: float = 0.0001,
    pred_shapley_weight: float = 0.0,
    bhvr_char_weight: float = 0.0,
    bhvr_shapley_weight: float = 0.0,
    behaviour_shapley_approx: bool = True,
    shapley_only: bool = False,
) -> Tuple[chex.Array, Tuple[Dict[str, chex.Array], optax.OptState]]:
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
    results = train_state.apply_fn(
        variables, x=experience.observation_nn, train=True, mutable=mutables
    )
    if mutables:
        preds, updates = results
    else:
        preds = results
        updates = {}

    pred_policy = preds["policy"]
    pred_value = preds["value"]

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

    # compute L2 regularization
    l2_reg = l2_reg_lambda * jax.tree_util.tree_reduce(
        lambda x, y: x + y, jax.tree_map(lambda x: (x**2).sum(), params)
    )

    # total loss
    loss = jnp.where(shapley_only, 0.0, policy_loss + value_loss) + l2_reg
    aux_metrics = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "l2_reg": l2_reg,
    }

    if pred_shapley_weight > 0.0:
        s_loss, s_metrics = compute_prediction_shapley_loss(
            params, train_state, experience.observation_nn, key
        )
        loss = loss + pred_shapley_weight * s_loss
        for k, v in s_metrics.items():
            aux_metrics[k] = v

    if bhvr_char_weight > 0.0:
        c_loss, c_metrics = compute_behaviour_characteristic_loss(
            params,
            train_state,
            experience.observation_nn,
            key,
            behaviour_shapley_approx,
        )
        loss = loss + bhvr_char_weight * c_loss
        for k, v in c_metrics.items():
            aux_metrics[k] = v

    if bhvr_shapley_weight > 0.0:
        bs_loss, bs_metrics = compute_behaviour_shapley_loss(
            params,
            train_state,
            experience.observation_nn,
            key,
            behaviour_shapley_approx,
        )
        loss = loss + bhvr_shapley_weight * bs_loss
        for k, v in bs_metrics.items():
            aux_metrics[k] = v

    return loss, (aux_metrics, updates)
