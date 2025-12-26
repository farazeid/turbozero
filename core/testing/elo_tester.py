import os
import random
from functools import partial
from typing import Dict, Optional, Tuple

import chex
import jax
import orbax.checkpoint as ocp

from core.common import two_player_game
from core.evaluators.evaluator import Evaluator
from core.testing.tester import BaseTester, TestState
from core.types import EnvInitFn, EnvStepFn


@chex.dataclass(frozen=True)
class RobustEloTesterState(TestState):
    best_params: chex.ArrayTree
    elo: float


class RobustEloTester(BaseTester):
    """
    Robust Elo Tester that evaluates agent against:
    1. The current 'best' model (to track progress/Elo).
    2. Random past models from a 'league' (to check robustness/forgetting).
    """

    def __init__(
        self,
        num_episodes: int,
        base_elo: float = 0.0,
        elo_k_factor: float = 40.0,
        min_win_rate_for_update: float = 0.55,
        league_dir: str = "/tmp/turbozero_checkpoints",
        *args,
        **kwargs,
    ):
        """
        Args:
        - `num_episodes`: number of episodes to evaluate
        - `base_elo`: starting Elo rating
        - `elo_k_factor`: K-factor for Elo updates
        - `min_win_rate_for_update`: win rate required to replace best model
        - `league_dir`: directory where checkpoints are stored
        """
        super().__init__(num_keys=num_episodes, *args, **kwargs)
        self.num_episodes = num_episodes
        self.base_elo = base_elo
        self.elo_k_factor = elo_k_factor
        self.min_win_rate_for_update = min_win_rate_for_update
        self.league_dir = league_dir

        # Setup basic checkpointer to read params
        self.checkpointer = ocp.PyTreeCheckpointer()

    def init(self, params: chex.ArrayTree, **kwargs) -> RobustEloTesterState:
        """Initializes the internal state of the Tester."""
        return RobustEloTesterState(
            best_params=params,  # Initialize best_params as current params
            elo=self.base_elo,
        )

    def _update_elo(self, current_elo: float, match_score: float) -> float:
        """updates elo rating based on match score against an equal opponent (best previous)."""
        expected_score = 0.5
        return current_elo + self.elo_k_factor * (match_score - expected_score)

    def _load_random_past_params(
        self, current_params: chex.ArrayTree
    ) -> Optional[chex.ArrayTree]:
        """Loads parameters from a random past checkpoint in the league directory."""
        if not os.path.exists(self.league_dir):
            return None

        try:
            items = os.listdir(self.league_dir)
            checkpoints = [
                int(item)
                for item in items
                if item.isdigit() and os.path.isdir(os.path.join(self.league_dir, item))
            ]
        except OSError:
            return None

        if not checkpoints:
            return None

        ckpt_epoch = random.choice(checkpoints)
        ckpt_path = os.path.join(self.league_dir, str(ckpt_epoch))

        try:
            # Attempt to restore params.
            # We assume structure matches what we expect or we rely on extracting 'params'.
            restored = self.checkpointer.restore(ckpt_path)

            if "params" in restored:
                return restored["params"]
            elif hasattr(restored, "params"):
                return restored.params
            else:
                return None

        except Exception as e:
            return None

    @partial(jax.pmap, axis_name="d", static_broadcasted_argnums=(0, 1, 2, 3, 4))
    def test_vs_params(
        self,
        max_steps: int,
        env_step_fn: EnvStepFn,
        env_init_fn: EnvInitFn,
        evaluator: Evaluator,
        keys: chex.PRNGKey,
        params_p1: chex.ArrayTree,
        params_p2: chex.ArrayTree,
    ) -> Tuple[chex.Array, Dict]:
        """Runs a match between two sets of parameters."""
        game_fn = partial(
            two_player_game,
            evaluator_1=evaluator,
            evaluator_2=evaluator,
            params_1=params_p1,
            params_2=params_p2,
            env_step_fn=env_step_fn,
            env_init_fn=env_init_fn,
            max_steps=max_steps,
        )

        results, _, _ = jax.vmap(game_fn)(keys)
        # Assuming result[0] is reward for p1. For {1, -1}, (r+1)/2 gives win rate.
        p1_avg_score = results[:, 0].mean()
        return p1_avg_score, {}

    def run(
        self,
        key: chex.PRNGKey,
        epoch_num: int,
        max_steps: int,
        num_devices: int,
        env_step_fn: EnvStepFn,
        env_init_fn: EnvInitFn,
        evaluator: Evaluator,
        state: RobustEloTesterState,
        params: chex.ArrayTree,
        *args,
    ) -> Tuple[RobustEloTesterState, Dict, str]:
        keys = self.split_keys(key, num_devices)

        # 1. Evaluate vs Best Params (Progress)
        vs_best_score_devices, _ = self.test_vs_params(
            max_steps,
            env_step_fn,
            env_init_fn,
            evaluator,
            keys,
            params,
            state.best_params,
        )
        vs_best_score = vs_best_score_devices.mean()
        win_rate_vs_best = (vs_best_score + 1) / 2

        new_elo = state.elo
        new_best_params = state.best_params

        if win_rate_vs_best > self.min_win_rate_for_update:
            new_elo = self._update_elo(state.elo, win_rate_vs_best)
            new_best_params = params

        metrics = {
            f"{self.name}_elo": new_elo,
            f"{self.name}_win_rate_vs_best": win_rate_vs_best,
        }

        # 2. Evaluate vs League (Robustness)
        past_params = self._load_random_past_params(params)

        if past_params is not None:
            # Basic shape check/protection could go here
            vs_past_score_devices, _ = self.test_vs_params(
                max_steps,
                env_step_fn,
                env_init_fn,
                evaluator,
                keys,
                params,
                past_params,
            )
            vs_past_score = vs_past_score_devices.mean()
            win_rate_vs_past = (vs_past_score + 1) / 2
            metrics[f"{self.name}_win_rate_vs_league"] = win_rate_vs_past

        new_state = state.replace(best_params=new_best_params, elo=new_elo)

        return new_state, metrics, None

    @partial(jax.pmap, axis_name="d", static_broadcasted_argnums=(0, 1, 2, 3, 4))
    def test(self, *args, **kwargs):
        raise NotImplementedError("Use run() for RobustEloTester")
