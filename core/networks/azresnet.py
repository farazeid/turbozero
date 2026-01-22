from dataclasses import dataclass

import flax.linen as nn
import jax


@dataclass
class AZResnetConfig:
    """Configuration for AlphaZero ResNet model:
    - `policy_head_out_size`: output size of the policy head (number of actions)
    - `num_blocks`: number of residual blocks
    - `num_channels`: number of channels in each residual block
    """

    policy_head_out_size: int
    num_blocks: int
    num_channels: int
    prediction_shapley_head: bool = False
    behaviour_characteristic_head: bool = False
    behaviour_shapley_head: bool = False
    behaviour_shapley_approx: bool = True
    include_policy_head: bool = True
    include_value_head: bool = True


class ResidualBlock(nn.Module):
    """Residual block for AlphaZero ResNet model.
    - `channels`: number of channels"""

    channels: int

    @nn.compact
    def __call__(self, x, train: bool):
        y = nn.Conv(
            features=self.channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)
        y = nn.BatchNorm(use_running_average=not train)(y)
        y = nn.relu(y)
        y = nn.Conv(
            features=self.channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(y)
        y = nn.BatchNorm(use_running_average=not train)(y)
        return nn.relu(x + y)


class AZResnet(nn.Module):
    """Implements the AlphaZero ResNet model.
    - `config`: network configuration"""

    config: AZResnetConfig

    @nn.compact
    def __call__(self, x, train: bool):
        # initial conv layer
        x = nn.Conv(
            features=self.config.num_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        # residual blocks
        for _ in range(self.config.num_blocks):
            x = ResidualBlock(channels=self.config.num_channels)(x, train=train)

        if self.config.include_policy_head:
            # policy head
            policy = nn.Conv(
                features=2,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="SAME",
                use_bias=False,
            )(x)
            policy = nn.BatchNorm(use_running_average=not train)(policy)
            policy = nn.relu(policy)
            spatial_policy_logits = policy
            policy = policy.reshape((policy.shape[0], -1))
            policy = nn.Dense(features=self.config.policy_head_out_size)(policy)
        else:
            policy = None
            spatial_policy_logits = None

        if self.config.include_value_head:
            # value head
            value = nn.Conv(
                features=1,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="SAME",
                use_bias=False,
            )(x)
            value = nn.BatchNorm(use_running_average=not train)(value)
            value = nn.relu(value)
            value = value.reshape((value.shape[0], -1))
            value = nn.Dense(features=1)(value)
            value = nn.tanh(value)
        else:
            value = None

        outputs = {}
        if policy is not None:
            outputs["policy"] = policy
            outputs["spatial_policy_logits"] = spatial_policy_logits
        if value is not None:
            outputs["value"] = value

        # Shared stop_gradient to prevent explanation heads from affecting backbone
        x_expl = jax.lax.stop_gradient(x)

        if self.config.prediction_shapley_head:
            # Predicts contribution of each spatial position to the value function
            outputs["prediction_shapley"] = nn.Conv(
                features=1,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="SAME",
                use_bias=True,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
                name="prediction_shapley_head",
            )(x_expl)

        if self.config.behaviour_characteristic_head:
            # Predicts policy distribution under observation masking
            out_features = (
                2
                if self.config.behaviour_shapley_approx
                else self.config.policy_head_out_size
            )
            outputs["behaviour_characteristic"] = nn.Conv(
                features=out_features,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="SAME",
                use_bias=True,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
                name="behaviour_characteristic_head",
            )(x_expl)

        if self.config.behaviour_shapley_head:
            # Predicts Shapley values for the policy distribution
            out_features = (
                2
                if self.config.behaviour_shapley_approx
                else self.config.policy_head_out_size
            )
            outputs["behaviour_shapley"] = nn.Conv(
                features=out_features,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="SAME",
                use_bias=True,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
                name="behaviour_shapley_head",
            )(x_expl)

        return outputs


@dataclass
class AZResnetSeparate(nn.Module):
    """Container for separate agent and explanation networks.
    - `agent`: The main AlphaZero agent network
    - `prediction_shapley_net`: Optional network for prediction shapley values
    - `behaviour_characteristic_net`: Optional network for behaviour characteristic values
    - `behaviour_shapley_net`: Optional network for behaviour shapley values
    """

    agent: AZResnet
    prediction_shapley_net: AZResnet | None = None
    behaviour_characteristic_net: AZResnet | None = None
    behaviour_shapley_net: AZResnet | None = None

    @nn.compact
    def __call__(self, x, train: bool):
        # Forward pass through the main agent
        outputs = self.agent(x, train=train)

        # Forward pass through auxiliary networks if they exist
        if self.prediction_shapley_net is not None:
            pred_out = self.prediction_shapley_net(x, train=train)
            outputs.update(pred_out)

        if self.behaviour_characteristic_net is not None:
            char_out = self.behaviour_characteristic_net(x, train=train)
            outputs.update(char_out)

        if self.behaviour_shapley_net is not None:
            shap_out = self.behaviour_shapley_net(x, train=train)
            outputs.update(shap_out)

        return outputs
