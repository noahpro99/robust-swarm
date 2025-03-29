import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import ray
from config import get_algo_config
from env import CustomEnv
from ray.tune.registry import register_env


def run_config(env_config, checkpoint_path, num_trials):
    env_config["train"] = False

    register_env("custom_env", lambda config: CustomEnv(config))

    algo_config = get_algo_config(env_config, train=False, num_trials=num_trials)

    algo = algo_config.build()
    algo.restore(checkpoint_path)

    eval_result = algo.evaluate()
    algo.stop()

    return eval_result["env_runners"]["hist_stats"]


def plot_all_param_results_grouped(all_param_results_path):
    with open(all_param_results_path, "r") as f:
        data = json.load(f)

        structured_data = {
            "steps": {},
            "friendly_reward_per_agent": {},
            "prey_reward_per_agent": {},
        }

    # Fill the new data structure based on the content
    for item in data:
        param_name = item[0]
        param_value = item[1]
        results = item[2]

        key = f"{param_name}={param_value}"
        structured_data["steps"][key] = results.get("episode_lengths", [])
        structured_data["friendly_reward_per_agent"][key] = results.get(
            "policy_friendly_policy_reward", []
        )
        structured_data["prey_reward_per_agent"][key] = results.get(
            "policy_prey_policy_reward", []
        )

    def plot_histograms(data_dict, x_label, y_label, title_prefix):
        unique_params = sorted(set(key.split("=")[0] for key in data_dict.keys()))
        num_params = len(unique_params)
        colors = plt.cm.viridis(np.linspace(0, 1, 3))  # type: ignore # Use a colormap to get distinct colors

        fig, axs = plt.subplots(
            (num_params + 1) // 2, 2, figsize=(16, 6 * ((num_params + 1) // 2))
        )
        fig.suptitle(f"{title_prefix} Analysis", fontsize=16)

        for ax, param in zip(axs.flatten(), unique_params):
            param_values = {
                key.split("=")[1]: values
                for key, values in data_dict.items()
                if key.startswith(param)
            }
            sorted_keys = sorted(param_values.keys(), key=lambda x: float(x))
            for color, key in zip(colors, sorted_keys):
                values = param_values[key]
                ax.hist(
                    values,
                    bins=50,
                    histtype="step",
                    color=color,
                    label=f"{param}={key}",
                )
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(param)
            ax.legend()

        # Remove any empty subplots
        for i in range(len(unique_params), len(axs.flatten())):
            fig.delaxes(axs.flatten()[i])

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.savefig(f"results/trial_results/{title_prefix}_analysis.png")

    # First overall plot: steps
    plot_histograms(structured_data["steps"], "Steps", "Density", "Steps")

    # Second overall plot: friendly_reward_per_agent
    plot_histograms(
        structured_data["friendly_reward_per_agent"],
        "Friendly Reward per Agent",
        "Density",
        "Friendly Reward per Agent",
    )

    # Third overall plot: prey_reward_per_agent
    plot_histograms(
        structured_data["prey_reward_per_agent"],
        "Prey Reward per Agent",
        "Density",
        "Prey Reward per Agent",
    )

    # plt.savefig(f"results/trial_results/{parameter}_analysis.png")


def run_trials(checkpoint_path, num_trials, base_config_path, test_config_path):
    # Load configs from JSON files
    with open(base_config_path, "r") as f:
        base_config = json.load(f)

    with open(test_config_path, "r") as f:
        test_config = json.load(f)

    os.makedirs("results/trial_results", exist_ok=True)

    ray.init(ignore_reinit_error=True)

    # Run trials with base config
    base_trial_result = run_config(base_config, checkpoint_path, num_trials)

    # Adjust each parameter in test_config and run trials
    all_param_results = []
    for param, adjustment in test_config.items():
        param_results = []
        param_results.append((param, base_config[param], base_trial_result))
        for direction in [-1, 1]:
            print(
                f"Running trials with {param}={base_config[param] + direction * adjustment}"
            )
            adjusted_config = base_config.copy()
            adjusted_config[param] += direction * adjustment
            trial_result = run_config(adjusted_config, checkpoint_path, num_trials)
            param_results.append((param, adjusted_config[param], trial_result))
        all_param_results.extend(param_results)

    ray.shutdown()

    # save to json
    with open("results/all_param_results.json", "w") as f:
        json.dump(all_param_results, f)


def main():
    parser = argparse.ArgumentParser(description="CLI to test the trained model")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    parser_run = subparsers.add_parser("run", help="Run the test trials")
    parser_run.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the checkpoint directory",
    )
    parser_run.add_argument(
        "-n",
        "--num_trials",
        type=int,
        default=10,
        help="Number of trials per configuration",
    )
    parser_run.add_argument(
        "--base_config_path",
        type=str,
        default="configs/config.json",
        help="Path to the base config JSON file",
    )
    parser_run.add_argument(
        "--test_config_path",
        type=str,
        default="configs/test_config.json",
        help="Path to the test config JSON file",
    )

    parser_plot = subparsers.add_parser("plot", help="Plot the results")
    parser_plot.add_argument(
        "--all_param_results_path",
        type=str,
        default="results/all_param_results.json",
        help="Path to the all param results JSON file",
    )

    args = parser.parse_args()
    if args.command == "run":
        run_trials(
            args.checkpoint_path,
            args.num_trials,
            args.base_config_path,
            args.test_config_path,
        )
    elif args.command == "plot":
        plot_all_param_results_grouped(args.all_param_results_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
