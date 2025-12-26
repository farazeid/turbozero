import os
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import optax
import pgx
import tyro

from core.evaluators.alphazero import AlphaZero
from core.evaluators.evaluation_fns import (
    make_nn_eval_fn,
    make_nn_eval_fn_no_params_callable,
)
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS
from core.memory.replay_memory import EpisodeReplayBuffer
from core.networks.azresnet import AZResnet, AZResnetConfig
from core.testing.two_player_baseline import TwoPlayerBaseline
from core.testing.utils import render_pgx_2p
from core.training.loss_fns import az_default_loss_fn
from core.training.train import Trainer
from core.types import StepMetadata


def step_fn(state, action):
    new_state = env.step(state, action)

    new_state = jax.tree_util.tree_map(
        lambda x: jnp.asarray(x) if isinstance(x, int) else x, new_state
    )

    return new_state, StepMetadata(
        rewards=new_state.rewards,
        action_mask=new_state.legal_action_mask,
        terminated=new_state.terminated,
        cur_player_id=new_state.current_player,
        step=new_state._step_count,
    )


def init_fn(key):
    state = env.init(key)

    state = jax.tree_util.tree_map(
        lambda x: jnp.asarray(x) if isinstance(x, int) else x, state
    )

    return state, StepMetadata(
        rewards=state.rewards,
        action_mask=state.legal_action_mask,
        terminated=state.terminated,
        cur_player_id=state.current_player,
        step=state._step_count,
    )


def state_to_nn_input(state):
    return state.observation


def greedy_eval(obs):
    value = (obs[..., 0].sum() - obs[..., 1].sum()) / 64
    return jnp.ones((1, env.num_actions)), jnp.array([value])


def make_rot_transform_fn(amnt: int):
    def rot_transform_fn(mask, policy, state):
        action_ids = jnp.arange(
            65
        )  # 65 total actions, but only rotate the first 64! (65th is always do nothing action)
        # we only use state.observation, no need to update the rest of the state fields
        new_obs = jnp.rot90(state.observation, amnt, axes=(-3, -2))
        # map action ids to new action ids
        idxs = jnp.arange(64).reshape(8, 8)  # rotate first 64 actions
        new_idxs = jnp.rot90(idxs, amnt, axes=(0, 1)).flatten()
        action_ids = action_ids.at[:64].set(new_idxs)
        # get new mask and policy
        new_mask = mask[..., action_ids]
        new_policy = policy[..., action_ids]
        return new_mask, new_policy, state.replace(observation=new_obs)

    return rot_transform_fn


@dataclass
class Args:
    resnet_num_blocks: int = 4
    resnet_num_channels: int = 32

    eval_num_iterations: int = 32
    eval_max_nodes: int = 40
    eval_temperature: float = 1.0

    eval_test_num_iterations: int = 64
    eval_test_max_nodes: int = 80
    eval_test_temperature: float = 0.0

    buffer_capacity: int = 1_000

    render_duration: int = 900

    batch_size: int = 1024
    train_batch_size: int = 4096
    warmup_steps: int = 0
    collection_steps_per_epoch: int = 256
    train_steps_per_epoch: int = 64

    l2_reg_lambda: float = 0.0

    learning_rate: float = 1e-3

    max_episode_steps: int = 80

    tester_num_episodes: int = 128

    seed: int = 0
    num_epochs: int = 100
    eval_every: int = 5


if __name__ == "__main__":
    os.environ["CUDA_PATH"] = "/usr/local/cuda"

    args = tyro.cli(Args)

    env = pgx.make("othello")

    resnet = AZResnet(
        AZResnetConfig(
            policy_head_out_size=env.num_actions,
            num_blocks=args.resnet_num_blocks,
            num_channels=args.resnet_num_channels,
        )
    )

    az_evaluator = AlphaZero(MCTS)(  # type: ignore[operator]
        eval_fn=make_nn_eval_fn(resnet, state_to_nn_input),
        num_iterations=args.eval_num_iterations,
        max_nodes=args.eval_max_nodes,
        branching_factor=env.num_actions,
        action_selector=PUCTSelector(),
        temperature=args.eval_temperature,
    )

    az_evaluator_test = AlphaZero(MCTS)(  # type: ignore[operator]
        eval_fn=make_nn_eval_fn(resnet, state_to_nn_input),
        num_iterations=args.eval_test_num_iterations,
        max_nodes=args.eval_test_max_nodes,
        branching_factor=env.num_actions,
        action_selector=PUCTSelector(),
        temperature=args.eval_test_temperature,
    )

    model = pgx.make_baseline_model("othello_v0")

    baseline_eval_fn = make_nn_eval_fn_no_params_callable(model, state_to_nn_input)

    baseline_az = AlphaZero(MCTS)(  # type: ignore[operator]
        eval_fn=baseline_eval_fn,
        num_iterations=args.eval_test_num_iterations,
        max_nodes=args.eval_test_max_nodes,
        branching_factor=env.num_actions,
        action_selector=PUCTSelector(),
        temperature=args.eval_test_temperature,
    )

    greedy_baseline_eval_fn = make_nn_eval_fn_no_params_callable(
        greedy_eval, state_to_nn_input
    )

    greedy_az = AlphaZero(MCTS)(  # type: ignore[operator]
        eval_fn=greedy_baseline_eval_fn,
        num_iterations=args.eval_test_num_iterations,
        max_nodes=args.eval_test_max_nodes,
        branching_factor=env.num_actions,
        action_selector=PUCTSelector(),
        temperature=args.eval_test_temperature,
    )

    replay_memory = EpisodeReplayBuffer(capacity=args.buffer_capacity)

    transforms = [make_rot_transform_fn(i) for i in range(1, 4)]

    render_fn = partial(
        render_pgx_2p,
        p1_label="Black",
        p2_label="White",
        duration=args.render_duration,
    )

    trainer = Trainer(
        batch_size=args.batch_size,
        train_batch_size=args.train_batch_size,
        warmup_steps=args.warmup_steps,
        collection_steps_per_epoch=args.collection_steps_per_epoch,
        train_steps_per_epoch=args.train_steps_per_epoch,
        nn=resnet,
        loss_fn=partial(az_default_loss_fn, l2_reg_lambda=args.l2_reg_lambda),
        optimizer=optax.adam(args.learning_rate),
        evaluator=az_evaluator,
        memory_buffer=replay_memory,
        max_episode_steps=args.max_episode_steps,
        env_step_fn=step_fn,
        env_init_fn=init_fn,
        state_to_nn_input_fn=state_to_nn_input,
        testers=[
            TwoPlayerBaseline(
                num_episodes=args.tester_num_episodes,
                baseline_evaluator=baseline_az,
                render_fn=render_fn,
                render_dir=".",
                name="pretrained",
            ),
            TwoPlayerBaseline(
                num_episodes=args.tester_num_episodes,
                baseline_evaluator=greedy_az,
                render_fn=render_fn,
                render_dir=".",
                name="greedy",
            ),
        ],
        evaluator_test=az_evaluator_test,
        data_transform_fns=transforms,
        wandb_entity="fastsverl",
        wandb_project_name="dev-go",
    )

    output = trainer.train_loop(seed=0, num_epochs=100, eval_every=5)
