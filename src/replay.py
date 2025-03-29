import argparse
import json

import ray
from config import get_algo_config
from env import CustomEnv
from ray.rllib.algorithms import Algorithm
from ray.tune.registry import register_env
from tqdm import tqdm


def render_episode(
    env: CustomEnv,
    algo: Algorithm,
) -> str:
    obs, _ = env.reset()
    done = {"__all__": False}
    total_rewards = {
        "friendly": 0,
        "prey": 0,
    }
    prog_bar = tqdm(total=env.max_steps)

    while not done["__all__"]:
        actions = {
            agent_id: algo.compute_single_action(
                obs[agent_id],
                policy_id=(
                    "friendly_policy" if "friendly" in agent_id else "prey_policy"
                ),
            )
            for agent_id in obs
        }
        obs, rewards, done, truncated, info = env.step(actions)
        total_rewards["friendly"] += sum([
            reward for agent_id, reward in rewards.items() if "friendly" in agent_id
        ])
        total_rewards["prey"] += sum([
            reward for agent_id, reward in rewards.items() if "prey" in agent_id
        ])
        env.render(obs=obs)
        prog_bar.update(1)

    prog_bar.close()
    print(f"Total rewards: {total_rewards}")

    # Create video from frames
    return env.create_video()


def main(checkpoint_path: str, config_path: str, num_episodes: int):
    with open(config_path) as f:
        env_config = json.load(f)
    register_env("custom_env", lambda config: CustomEnv(config))

    ray.init(ignore_reinit_error=True)

    config = get_algo_config(env_config, train=False)

    algo = config.build()

    algo.restore(checkpoint_path)

    env = CustomEnv(env_config)

    for _ in range(num_episodes):
        episode_path = render_episode(env, algo)
        print(f"Episode saved at: {episode_path}")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render episodes of the environment using a trained algorithm."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/config.json",
        help="Path to the env configuration file",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the checkpoint directory",
    )
    parser.add_argument(
        "-n",
        "--num_episodes",
        type=int,
        default=1,
        help="Number of episodes to run",
    )
    args = parser.parse_args()
    main(args.checkpoint_path, args.config_path, args.num_episodes)
