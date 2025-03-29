import os
from datetime import datetime

import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from ray.rllib.env.multi_agent_env import MultiAgentEnv


def get_agent_ids(num_friendlies: int, num_prey: int) -> list[str]:
    return [f"friendly_{i}" for i in range(num_friendlies)] + [
        f"prey_{i}" for i in range(num_prey)
    ]


def get_observation_space(num_friendlies: int, num_prey: int) -> gym.spaces.Space:
    agent_ids = get_agent_ids(num_friendlies, num_prey)

    observation_space = gym.spaces.Box(
        low=-1e3, high=1e3, shape=(len(agent_ids) * 6 + 2,), dtype=np.float32
    )
    return observation_space


def get_action_space(num_friendlies: int, num_prey: int) -> gym.spaces.Space:
    return gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)


class CustomEnv(MultiAgentEnv):
    def __init__(self, config: dict):
        self.num_friendlies: int = config.get("num_friendlies", 4)
        self.num_prey: int = config.get("num_prey", 1)
        self.scare_arm_length: float = config.get("scare_arm_length", 3.0)
        self.collide_radius: float = config.get("collide_radius", 0.5)
        self.prey_speed: float = config.get("prey_speed", 1.0)
        self.friendly_speed: float = config.get("friendly_speed", 1.0)
        self.noise_distance: float = config.get("noise_distance", 0.1)
        self.noise_direction: float = config.get("noise_direction", 0.05)
        self.update_mean: float = config.get("update_mean", 3)
        self.update_std: float = config.get("update_std", 0.8)
        self.prey_win_distance: float = config.get("prey_win_distance", 45)
        self.max_steps: int = config.get("max_steps", 200)
        self.training: bool = config.get("train", True)

        self.action_space = get_action_space(self.num_friendlies, self.num_prey)
        self.observation_space = get_observation_space(
            self.num_friendlies, self.num_prey
        )

        self._agent_ids = get_agent_ids(self.num_friendlies, self.num_prey)
        os.makedirs("frames", exist_ok=True)
        os.makedirs("episodes", exist_ok=True)
        super().__init__()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.rnjesus = np.random.default_rng(seed=seed)
        self.step_count: int = 0

        # Initialize the mean position for the agents
        agent_theta = self.rnjesus.uniform(0, 2 * np.pi)
        mean_distance = 15
        agent_mean_start = np.array([
            mean_distance * np.cos(agent_theta),
            mean_distance * np.sin(agent_theta),
        ])

        agent_dist_from_mean = 5

        # Initialize agent positions around the mean start position
        self.agent_positions: dict[str, np.ndarray] = {
            f"friendly_{i}": agent_mean_start
            + agent_dist_from_mean * np.array([np.cos(theta), np.sin(theta)])
            for i, theta in enumerate(
                np.linspace(0, 2 * np.pi, self.num_friendlies, endpoint=False)
            )
        }
        # put the prey around 0,0
        prey_dist_from_mean = 2
        for i, theta in enumerate(
            np.linspace(0, 2 * np.pi, self.num_prey, endpoint=False)
        ):
            self.agent_positions[f"prey_{i}"] = prey_dist_from_mean * np.array([
                np.cos(theta),
                np.sin(theta),
            ])

        self.prey_immobilized = {f"prey_{i}": False for i in range(self.num_prey)}

        self.steps_until_obs_update: dict[str, dict[str, int]] = {
            id: {other_id: 0 for other_id in self.agent_positions.keys()}
            for id in self.agent_positions.keys()
        }

        self.prev_obs = {
            id: np.zeros((len(self._agent_ids) * 6 + 2), dtype=np.float32)
            for id in self._agent_ids
        }

        info = (
            {
                "initial_angle": agent_theta,
            }
            if not self.training
            else {}
        )
        obs = self._get_obs()
        return obs, info

    def step(self, action_dict):
        self.step_count += 1
        rewards = {agent_id: 0.0 for agent_id in action_dict}
        done = {"__all__": False}
        truncated = {"__all__": False}

        # move agents
        for agent_id, action in action_dict.items():
            speed = (
                self.friendly_speed
                if "friendly" in agent_id
                else self.prey_speed
                if not self.prey_immobilized[agent_id]
                else 0
            )
            action = speed * np.clip(np.array(action), -1, 1)
            self.agent_positions[agent_id] += action

        for prey_id, prey_pos in [
            (id, pos) for id, pos in self.agent_positions.items() if "prey" in id
        ]:
            crashed = any(
                np.linalg.norm(prey_pos - other_pos) < self.collide_radius * 2
                for other_id, other_pos in self.agent_positions.items()
                if prey_id != other_id and "friendly" in other_id
            )
            if not self._prey_can_move(prey_pos) and not crashed:
                self.prey_immobilized[prey_id] = True

        info = {
            agent_id: {"position": self.agent_positions[agent_id]}
            for agent_id in action_dict
        }

        # if all prey immobilized or prey gets away end the episode
        if all(self.prey_immobilized.values()):
            done["__all__"] = True

        # if the max steps are reached end the episode
        if self.step_count >= self.max_steps:
            done["__all__"] = True
            truncated["__all__"] = True

        # calculate rewards
        for agent_id, agent_position in self.agent_positions.items():
            # punish agents for colliding with other agents
            for other_agent_id, other_position in self.agent_positions.items():
                if (
                    agent_id != other_agent_id
                    and np.linalg.norm(agent_position - other_position)
                    < self.collide_radius * 2
                ):
                    rewards[agent_id] -= 0.01

            if "friendly" in agent_id:
                # small reward for getting closer to the closest prey not immobilized
                nearest_prey_position = self._find_nearest_non_immobile_prey_position(
                    agent_id
                )
                if nearest_prey_position is not None:
                    rewards[agent_id] -= (
                        0.0001
                        * np.linalg.norm(agent_position - nearest_prey_position).item()
                    )

                # even smaller reward for getting further mean distance to other agents
                rewards[agent_id] -= (
                    0.00001
                    * np.mean([
                        np.linalg.norm(agent_position - other_position)
                        for other_id, other_position in self.agent_positions.items()
                        if "friendly" in other_id and other_id != agent_id
                    ]).item()
                )

                if done["__all__"]:
                    # get penalty based on how many steps the friendly survived
                    rewards[agent_id] -= self.step_count / self.max_steps

            if "prey" in agent_id:
                if done["__all__"]:
                    # get reward based on how many steps the prey survived
                    rewards[agent_id] += self.step_count / self.max_steps

        obs = self._get_obs()
        rewards = {k: np.nan_to_num(v) for k, v in rewards.items()}
        return obs, rewards, done, truncated, info

    def _prey_can_move(self, prey_pos: np.ndarray):
        for theta in np.linspace(0, 2 * np.pi, 360):
            if all(
                not self._prey_ray_intersects_friendly_scare_arms(
                    prey_pos, friendly_pos, theta
                )
                for friendly_id, friendly_pos in self.agent_positions.items()
                if "friendly" in friendly_id
            ):
                return True

        return False

    def _prey_ray_intersects_friendly_scare_arms(
        self, prey_pos: np.ndarray, friendly_pos: np.ndarray, direction: float
    ):
        theta_to_friendly = np.arctan2(
            friendly_pos[1] - prey_pos[1],
            friendly_pos[0] - prey_pos[0],
        )
        theta_diff = np.abs(theta_to_friendly - direction) % (2 * np.pi)

        if theta_diff > np.pi:
            theta_diff = 2 * np.pi - theta_diff

        if theta_diff > np.pi / 2:
            return False

        opposite = np.tan(theta_diff) * np.linalg.norm(prey_pos - friendly_pos)

        return opposite < self.scare_arm_length

    def _get_obs(self, seed=None):
        obs = {id: self.prev_obs[id].copy() for id in self.prev_obs.keys()}

        for i, (agent_id, agent_position) in enumerate(self.agent_positions.items()):
            # set last two values to the position of the agent
            obs[agent_id][-2:] = agent_position

            for j, (other_entity_id, other_entity_position) in enumerate(
                self.agent_positions.items()
            ):
                # 0-1: relative position
                # 2-3: previous relative position
                # 4: agent type
                # 5: obs age

                # obs[agent_id][other_entity_id]["obs_age"] += 1
                obs[agent_id][j * 6 + 5] += 1

                self.steps_until_obs_update[agent_id][other_entity_id] -= 1

                if self.steps_until_obs_update[agent_id][other_entity_id] <= 0:
                    relative_position = other_entity_position - agent_position
                    distance = np.linalg.norm(relative_position)

                    noisy_theta = np.arctan2(
                        relative_position[1], relative_position[0]
                    ) + self.rnjesus.normal(0, self.noise_direction)

                    noisy_distance = (
                        distance
                        + self.rnjesus.normal(0, self.noise_distance) * distance
                    )
                    noisy_position = np.array(
                        [
                            noisy_distance * np.cos(noisy_theta),
                            noisy_distance * np.sin(noisy_theta),
                        ],
                        dtype=np.float32,
                    )

                    # obs[agent_id][other_entity_id]["prev_agent_rel_pos"] = obs[
                    #     agent_id
                    # ][other_entity_id]["agent_rel_pos"].copy()
                    obs[agent_id][j * 6 + 2 : j * 6 + 4] = obs[agent_id][
                        j * 6 : j * 6 + 2
                    ]

                    # obs[agent_id][other_entity_id]["agent_rel_pos"] = noisy_position
                    obs[agent_id][j * 6 : j * 6 + 2] = noisy_position
                    # obs[agent_id][other_entity_id]["agent_type"] = (
                    #     0 if "friendly" in other_entity_id else 1
                    # )
                    obs[agent_id][j * 6 + 4] = 0 if "friendly" in other_entity_id else 1

                    # obs[agent_id][other_entity_id]["obs_age"] = 0
                    obs[agent_id][j * 6 + 5] = 0

                    self.steps_until_obs_update[agent_id][other_entity_id] = int(
                        self.rnjesus.normal(self.update_mean, self.update_std)
                    )

        self.prev_obs = {id: obs[id].copy() for id in obs.keys()}

        return obs

    def _find_nearest_non_immobile_prey_position(
        self, agent_id: str
    ) -> np.ndarray | None:
        agent_position = self.agent_positions[agent_id]
        prey_positions = [
            pos
            for id, pos in self.agent_positions.items()
            if "prey" in id and not self.prey_immobilized[id]
        ]
        if not prey_positions:
            return None
        return min(prey_positions, key=lambda pos: np.linalg.norm(agent_position - pos))

    def render(self, mode="human", obs: dict[str, np.ndarray] | None = None):
        plt.figure(figsize=(8, 8))
        for agent_id, agent_position in self.agent_positions.items():
            if "friendly" in agent_id:
                if obs is not None:
                    # print small grey dot for all observations of other agents
                    for j, (other_entity_id, other_entity_position) in enumerate(
                        self.agent_positions.items()
                    ):
                        if other_entity_id != agent_id:
                            plt.plot(
                                obs[agent_id][j * 6] + agent_position[0],
                                obs[agent_id][j * 6 + 1] + agent_position[1],
                                "o",
                                color=(
                                    "gray" if obs[agent_id][j * 6 + 4] == 0 else "black"
                                ),
                                markersize=2,
                            )

                nearest_prey_position = self._find_nearest_non_immobile_prey_position(
                    agent_id
                )
                if nearest_prey_position is not None:
                    # plot scare arms as thin green dashed line
                    # perpendicular to the agent's direction to the prey
                    direction = nearest_prey_position - agent_position
                    direction /= np.linalg.norm(direction)
                    arm = (
                        np.array([-direction[1], direction[0]]) * self.scare_arm_length
                    )

                    line = Line2D(
                        [agent_position[0] - arm[0], agent_position[0] + arm[0]],
                        [agent_position[1] - arm[1], agent_position[1] + arm[1]],
                        color="green",
                        linestyle="--",
                    )
                    plt.gca().add_line(line)

                # plot collide radius as red solid circle
                circle = Circle(
                    tuple(agent_position), self.collide_radius, color="blue", fill=False
                )
                plt.gca().add_patch(circle)

            if "prey" in agent_id:
                # plot the prey's collision radius as a red solid circle
                circle = Circle(
                    tuple(agent_position), self.collide_radius, color="red", fill=False
                )
                plt.gca().add_patch(circle)

        # calculate max abs of agent positions to set the limits
        all_values = np.concatenate([
            list(pos) for pos in self.agent_positions.values()
        ])
        most_extreme_position_number = max(
            np.abs(all_values.min()), np.abs(all_values.max())
        )

        # make sure the most extreme position is at least 40
        most_extreme_position_number = max(40, most_extreme_position_number)
        buffer = 5

        plt.xlim(
            -most_extreme_position_number - buffer,
            most_extreme_position_number + buffer,
        )
        plt.ylim(
            -most_extreme_position_number - buffer,
            most_extreme_position_number + buffer,
        )

        # set the aspect ratio is equal
        plt.gca().set_aspect("equal", adjustable="box")
        plt.savefig(f"frames/frame_{self.step_count}.png")
        plt.close()

    def create_video(self) -> str:
        unique_id = datetime.now().strftime("%Y%m%d%H%M%S")
        frames = [f"frames/frame_{i}.png" for i in range(1, self.step_count + 1)]
        video_path = f"episodes/simulation_{unique_id}.mp4"
        writer = imageio.get_writer(video_path, fps=10, format="FFMPEG")  # type: ignore
        for frame in frames:
            image = imageio.imread(frame)
            writer.append_data(image)
        writer.close()
        for frame in frames:
            os.remove(frame)
        return video_path
