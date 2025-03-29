import argparse
import json
import os
import time

import ray
from config import get_algo_config
from env import CustomEnv
from ray.tune.registry import register_env


def main(config_path: str, checkpoint_dir: str | None = None, checkpoint_interval: int = 10):
    with open(config_path) as f:
        env_config = json.load(f)

    register_env("custom_env", lambda config: CustomEnv(config))

    ray.init(ignore_reinit_error=True)
    config = get_algo_config(env_config)
    algo = config.build()

    if checkpoint_dir:
        algo.restore(checkpoint_dir)
    else:
        logdir = "./ray_results"
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        checkpoint_dir = os.path.join(logdir, f"checkpoints_{timestamp}")
        os.makedirs(checkpoint_dir, exist_ok=True)

    total_time = 0
    i = 0
    num_failed_in_a_row = 0

    while True:
        start_time = time.time()

        try:
            result = algo.train()
        except Exception as e:
            print(f"Error at iteration {i}: {e}")
            algo.restore(checkpoint_dir)
            num_failed_in_a_row += 1
            time.sleep(num_failed_in_a_row * 5)
            if num_failed_in_a_row > 5:
                print(f"Too many failures in a row, exiting on iteration {i}")
                break
            continue
        num_failed_in_a_row = 0

        iteration_time = time.time() - start_time
        total_time += iteration_time

        print(
            f"{i} {iteration_time:.1f}s ",
            f"friendly: {result['env_runners']['policy_reward_mean'].get('friendly_policy', 0):.2f} "        )

        if i % checkpoint_interval == 0 and i != 0:
            algo.save(checkpoint_dir)
            print(f"Checkpoint saved at iteration {i} in {checkpoint_dir}")
        i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/config.json",
        help="Path to the env configuration file",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_dir",
        type=str,
        required=False,
        help="Path to the checkpoint directory",
    )

    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=10,
        help="Interval (in iterations) at which to save checkpoints",
    )
    args = parser.parse_args()

    main(args.config_path, args.checkpoint_dir, args.checkpoint_interval)
