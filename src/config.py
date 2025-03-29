from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

from env import get_action_space, get_observation_space


def policy_map_fn(agent_id: str, _episode=None, _worker=None, **_kwargs) -> str:
    if "friendly" in agent_id:
        return "friendly_policy"
    else:
        raise RuntimeError(f"Invalid agent_id: {agent_id}")


def get_multiagent_policies(num_friendlies: int) -> dict[str, PolicySpec]:
    policies: dict[str, PolicySpec] = {}

    observation_space = get_observation_space(num_friendlies=num_friendlies)

    action_space = get_action_space(num_friendlies=num_friendlies)

    policies["friendly_policy"] = PolicySpec(
        policy_class=None,
        observation_space=observation_space,
        action_space=action_space,
        config={},
    )

    policies["prey_policy"] = PolicySpec(
        policy_class=None,
        observation_space=observation_space,
        action_space=action_space,
        config={},
    )

    return policies


def get_algo_config(env_config: dict, train: bool = True, num_trials: int = 20):
    policies = get_multiagent_policies(
        num_friendlies=env_config["num_friendlies"],
    )

    num_gpus = 1
    model_shape = [32, 64, 32]
    model_activation = "tanh"

    config = (
        PPOConfig()
        .environment("custom_env", env_config=env_config)
        .framework("torch")
        .resources(num_gpus=num_gpus)
        .env_runners(
            num_env_runners=54,
            num_envs_per_env_runner=10,
            rollout_fragment_length=200,
            num_cpus_per_env_runner=1,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_map_fn,
            policies_to_train=list(policies.keys()),
        )
        .training(
            model={
                "fcnet_hiddens": model_shape,
                "fcnet_activation": model_activation,
            },
            train_batch_size=54 * 10 * 200,
            sgd_minibatch_size=54 * 10 * 200 // 54 * 11,  # type: ignore
            lr=1e-4,
        )
    )

    if not train:
        config = (
            PPOConfig()
            .environment("custom_env", env_config=env_config)
            .framework("torch")
            .resources(num_gpus=num_gpus)
            .multi_agent(
                policies=policies,
                policy_mapping_fn=policy_map_fn,
                policies_to_train=list(policies.keys()),
            )
            .training(
                model={
                    "fcnet_hiddens": model_shape,
                    "fcnet_activation": model_activation,
                },
            )
            .evaluation(
                evaluation_interval=1,
                evaluation_duration=num_trials,
                evaluation_num_env_runners=54,
            )
        )

    return config
