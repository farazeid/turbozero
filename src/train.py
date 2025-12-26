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
from core.evaluators.mcts.action_selection import MuZeroPUCTSelector
from core.evaluators.mcts.mcts import MCTS
from core.memory.replay_memory import EpisodeReplayBuffer
from core.networks.azresnet import AZResnet, AZResnetConfig
from core.testing.elo_tester import RobustEloTester
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
    num_board_pos = obs.shape[-3] * obs.shape[-2]
    value = (obs[..., 0].sum() - obs[..., 1].sum()) / num_board_pos
    return jnp.ones((1, env.num_actions)), jnp.array([value])


def make_rot_transform_fn(amnt: int):
    def rot_transform_fn(mask, policy, state):
        num_board_pos = env.num_actions - 1
        board_length = int(num_board_pos**0.5)
        action_ids = jnp.arange(
            env.num_actions
        )  # total actions, but only rotate the spatial ones! (last is always do nothing/pass action)
        # we only use state.observation, no need to update the rest of the state fields
        new_obs = jnp.rot90(state.observation, amnt, axes=(-3, -2))
        # map action ids to new action ids
        idxs = jnp.arange(num_board_pos).reshape(
            board_length, board_length
        )  # rotate spatial actions
        new_idxs = jnp.rot90(idxs, amnt, axes=(0, 1)).flatten()
        action_ids = action_ids.at[:num_board_pos].set(new_idxs)
        # get new mask and policy
        new_mask = mask[..., action_ids]
        new_policy = policy[..., action_ids]
        return new_mask, new_policy, state.replace(observation=new_obs)

    return rot_transform_fn


@dataclass
class Args:
    resnet_num_blocks: int = 20
    resnet_num_channels: int = 256

    eval_num_iterations: int = 800
    eval_max_nodes: int = 1200
    eval_temperature: float = 1.0

    eval_test_num_iterations: int = 800
    eval_test_max_nodes: int = 1200
    eval_test_temperature: float = 0.0

    puct_c1: float = 1.25
    puct_c2: float = 19652

    buffer_capacity: int = 1_000_000
    ckpt_dir: str = "/tmp/turbozero_checkpoints"

    render_duration: int = 900

    batch_size: int = 2048
    train_batch_size: int = 2048
    warmup_steps: int = 0
    collection_steps_per_epoch: int = 2048
    train_steps_per_epoch: int = 1000

    l2_reg_lambda: float = 1e-4

    learning_rate_0: float = 1e-2
    learning_rate_1: float = 1e-3
    learning_rate_2: float = 1e-4

    max_episode_steps: int = 700  # Go games are longer

    tester_num_episodes: int = 128
    elo_eval_num_episodes: int = 128
    elo_base: float = 0.0
    min_win_rate_elo: float = 0.55

    seed: int = 0
    num_epochs: int = 700
    eval_every: int = 5


if __name__ == "__main__":
    os.environ["CUDA_PATH"] = "/usr/local/cuda"

    args = tyro.cli(Args)

    env = pgx.make("go_19x19")

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
        action_selector=MuZeroPUCTSelector(c1=args.puct_c1, c2=args.puct_c2),
        temperature=args.eval_temperature,
    )

    az_evaluator_test = AlphaZero(MCTS)(  # type: ignore[operator]
        eval_fn=make_nn_eval_fn(resnet, state_to_nn_input),
        num_iterations=args.eval_test_num_iterations,
        max_nodes=args.eval_test_max_nodes,
        branching_factor=env.num_actions,
        action_selector=MuZeroPUCTSelector(c1=args.puct_c1, c2=args.puct_c2),
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
        action_selector=MuZeroPUCTSelector(c1=args.puct_c1, c2=args.puct_c2),
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
        optimizer=optax.sgd(
            learning_rate=optax.piecewise_constant_schedule(
                init_value=args.learning_rate_0,
                boundaries_and_scales={
                    200_000: args.learning_rate_1,
                    400_000: args.learning_rate_2,
                },
            ),
            momentum=0.9,
        ),
        evaluator=az_evaluator,
        memory_buffer=replay_memory,
        max_episode_steps=args.max_episode_steps,
        env_step_fn=step_fn,
        env_init_fn=init_fn,
        state_to_nn_input_fn=state_to_nn_input,
        testers=[
            TwoPlayerBaseline(
                num_episodes=args.tester_num_episodes,
                baseline_evaluator=greedy_az,
                render_fn=render_fn,
                render_dir=".",
                name="greedy",
            ),
            RobustEloTester(
                num_episodes=args.elo_eval_num_episodes,
                base_elo=args.elo_base,
                min_win_rate_for_update=args.min_win_rate_elo,
                render_fn=render_fn,
                render_dir=".",
                league_dir=args.ckpt_dir,
                name="selfplay",
            ),
        ],
        evaluator_test=az_evaluator_test,
        data_transform_fns=transforms,
        wandb_entity="fastsverl",
        wandb_project_name="dev-go",
        ckpt_dir=args.ckpt_dir,
    )

    output = trainer.train_loop(
        seed=args.seed,
        num_epochs=args.num_epochs,
        eval_every=args.eval_every,
    )
