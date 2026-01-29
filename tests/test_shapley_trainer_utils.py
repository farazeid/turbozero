import os
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import tyro

import wandb
from core.networks.katago import KataGoConfig, KataGoNetwork
from core.networks.shapley import BehaviourShapley, ShapleyConfig
from core.training.shapley_trainer import ShapleyTrainer


@dataclass
class Args:
    batch_size: int = 2
    pos_len: int = 19
    num_features: int = 22
    num_blocks: int = 2
    num_channels: int = 32
    num_mid_channels: int = 32
    c_gpool: int = 16
    lr: float = 1e-3
    wandb_project: str = "fastsverl-tests"
    wandb_entity: str = "fastsverl"
    wandb_name: str = Path(__file__).stem


def test_shapley_train_step_integration(args: Args):
    # Initialize W&B
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY not found in environment.")

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        mode="online",
    )

    # Setup
    key = jax.random.PRNGKey(0)
    H, W, C = args.pos_len, args.pos_len, args.num_features

    # 1. Initialize Real Models
    # Use small configs for speed
    agent_config = KataGoConfig(
        num_blocks=args.num_blocks,
        num_channels=args.num_channels,
        num_mid_channels=args.num_mid_channels,
        c_gpool=args.c_gpool,
    )
    agent = KataGoNetwork(config=agent_config)

    shapley_config = ShapleyConfig(
        num_blocks=args.num_blocks,
        num_channels=args.num_channels,
        num_mid_channels=args.num_mid_channels,
        multi_action=False,  # For simply testing single outcome/behaviour scalar
    )
    # Testing BehaviourShapley (outputting 362 logits)
    shapley_model = BehaviourShapley(config=shapley_config, num_actions=362)

    # Init variables
    key, k1, k2 = jax.random.split(key, 3)
    dummy_input = jnp.zeros((args.batch_size, H, W, C))

    agent_vars = agent.init(k1, dummy_input, train=False)

    # 2. Initialize Trainer
    optimizer = optax.adam(args.lr)
    trainer = ShapleyTrainer(shapley_type="behaviour", optimizer=optimizer)

    train_state = trainer.create_train_state(k2, shapley_model, dummy_input)

    # 3. Run Train Step
    batch = {"binaryInputNCHW": jax.random.normal(key, (args.batch_size, H, W, C))}

    key, step_key = jax.random.split(key)
    new_state, metrics = trainer.train_step(
        train_state=train_state,
        agent_apply_fn=agent.apply,
        agent_variables=agent_vars,
        batch=batch,
        key=step_key,
    )

    # Assertions
    assert "loss" in metrics
    assert "shapley_loss" in metrics
    assert metrics["loss"] > 0

    # Log to W&B
    wandb.log(metrics)
    wandb.log({"success": True})
    print(f"Metrics: {metrics}")

    wandb.finish()


def test_shapley_types():
    # Test mapping of types
    # Mocking output tuple
    B = 2
    policy = jnp.zeros((B, 362))
    value = jnp.ones((B, 1))
    ownership = jnp.zeros((B, 19, 19, 1))
    score = jnp.zeros((B, 1))

    out = (policy, value, ownership, score)

    from core.training.shapley_trainer import get_agent_target

    t_behav = get_agent_target(out, "behaviour")
    assert t_behav.shape == (B, 362)
    assert jnp.allclose(t_behav, jax.nn.softmax(policy))

    t_out = get_agent_target(out, "outcome")
    assert t_out.shape == (B, 1)

    t_score = get_agent_target(out, "score")
    assert t_score.shape == (B, 1)


if __name__ == "__main__":
    args = tyro.cli(Args)
    test_shapley_train_step_integration(args)
    test_shapley_types()
