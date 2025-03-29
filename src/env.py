import os
from datetime import datetime

import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from ray.rllib.env.multi_agent_env import MultiAgentEnv


def get_agent_ids(num_friendlies: int) -> list[str]:
    return [f"friendly_{i}" for i in range(num_friendlies)]


def get_observation_space(num_friendlies: int) -> gym.spaces.Space:
    agent_ids = get_agent_ids(num_friendlies)

    observation_space = gym.spaces.Dict(
        {
            "relative_agent_positions": gym.spaces.Box(
                low=-1e3, high=1e3, shape=(len(agent_ids) * 2,), dtype=np.float32
            ),
            "agents_alive": gym.spaces.MultiBinary(len(agent_ids)),
            "our_position": gym.spaces.Box(
                low=-1e3, high=1e3, shape=(2,), dtype=np.float32
            ),
            "relative_target_position": gym.spaces.Box(
                low=-1e3, high=1e3, shape=(2,), dtype=np.float32
            ),
        }
    )

    return observation_space


def get_action_space(num_friendlies: int) -> gym.spaces.Space:
    return gym.spaces.Dict(
        {
            "movement": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "send_stream": gym.spaces.Discrete(num_friendlies + 1),
        }
    )


class CustomEnv(MultiAgentEnv):
    def __init__(self, config: dict):
        self.num_friendlies: int = config.get("num_friendlies", 5)
        self.collide_radius: float = config.get("collide_radius", 0.5)
        self.friendly_speed: float = config.get("friendly_speed", 1.0)
        self.target_speed: float = config.get("target_speed", 0.7)
        self.optimal_target_dist = config.get("optimal_target_dist", 5.0)
        self.step_communication_down = config.get("step_communication_down", 100)
        self.num_drones_down = config.get("num_drones_down", 2)
        self.max_steps: int = config.get("max_steps", 200)
        self.training: bool = config.get("train", True)

        self.action_space = get_action_space(self.num_friendlies)
        self.observation_space = get_observation_space(self.num_friendlies)

        self._agent_ids = get_agent_ids(self.num_friendlies)
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
        agent_mean_start = np.array(
            [
                mean_distance * np.cos(agent_theta),
                mean_distance * np.sin(agent_theta),
            ]
        )

        agent_dist_from_mean = 5

        # Initialize agent positions around the mean start position
        self.agent_positions: dict[str, np.ndarray] = {
            f"friendly_{i}": agent_mean_start
            + agent_dist_from_mean * np.array([np.cos(theta), np.sin(theta)])
            for i, theta in enumerate(
                np.linspace(0, 2 * np.pi, self.num_friendlies, endpoint=False)
            )
        }
        self.agents_alive = {agent_id: True for agent_id in self.agent_positions}
        self.target_pos = np.array([-5, 5])

        obs = self._get_obs()
        return obs

    def step(self, action_dict):
        self.step_count += 1
        self._last_actions = action_dict
        rewards = {agent_id: 0.0 for agent_id in action_dict}
        done = {"__all__": False}
        truncated = {"__all__": False}

        # move agents
        for agent_id, action in action_dict.items():
            action = np.multiply(
                self.friendly_speed, np.clip(np.array(action["movement"]), -1, 1)
            )
            self.agent_positions[agent_id] += action
        # move target to the right
        self.target_pos[0] += self.target_speed

        info = {
            agent_id: {"position": self.agent_positions[agent_id]}
            for agent_id in action_dict
        }
        info["target"] = {"position": self.target_pos}

        # if the max steps are reached end the episode
        if self.step_count >= self.max_steps:
            done["__all__"] = True
            truncated["__all__"] = True

        # calculate rewards
        stream_reward = 0
        for agent_id, agent_position in self.agent_positions.items():
            # if agent is alive and steam_target is not index -1 then steam and get reward
            if (
                self.agents_alive[agent_id]
                and action_dict[agent_id]["send_stream"] != self.num_friendlies
            ):
                distance_to_target = np.linalg.norm(agent_position - self.target_pos)
                midpoint_tower_to_agent = agent_position / 2
                receiver_position = self.agent_positions[
                    f"friendly_{action_dict[agent_id]['send_stream']}"
                ]

                stream_reward += 0.5 * (
                    np.exp(-distance_to_target)
                    + np.exp(
                        -np.linalg.norm(receiver_position - midpoint_tower_to_agent)
                    )
                )

            # punish agents for colliding with other agents
            for other_agent_id, other_position in self.agent_positions.items():
                if (
                    agent_id != other_agent_id
                    and np.linalg.norm(agent_position - other_position)
                    < self.collide_radius * 2
                ):
                    rewards[agent_id] -= 0.01

        # give all agents the stream reward
        for agent_id in self.agent_positions:
            rewards[agent_id] += stream_reward

        obs = self._get_obs()
        rewards = {k: np.nan_to_num(v) for k, v in rewards.items()}
        return obs, rewards, done, truncated, info

    def _get_obs(self, seed=None):
        obs = {}

        for agent_id, agent_position in self.agent_positions.items():

            # convert agent positions to relative positions
            relative_positions = np.array(
                [
                    other_position - agent_position
                    for other_position in self.agent_positions.values()
                ]
            ).flatten()

            relative_target_position = self.target_pos - agent_position

            obs[agent_id] = {
                "relative_agent_positions": relative_positions,
                "agents_alive": np.array(
                    [
                        1 if self.agents_alive[agent_id] else 0
                        for agent_id in self.agent_positions
                    ]
                ),
                "our_position": agent_position,
                "relative_target_position": relative_target_position,
            }

        return obs

    def render(self, obs: dict[str, dict]):
        plt.figure(figsize=(8, 8))

        # Plot the tower at origin
        plt.plot(0, 0, "ks", markersize=10, label="Tower")

        # Plot target
        plt.plot(
            self.target_pos[0], self.target_pos[1], "r*", markersize=10, label="Target"
        )

        # Plot all agents
        for agent_id, agent_position in self.agent_positions.items():
            # Plot agent
            color = "blue" if self.agents_alive[agent_id] else "gray"
            plt.plot(
                agent_position[0], agent_position[1], "o", color=color, markersize=6
            )

            # Plot collision radius
            circle = Circle(
                tuple(agent_position), self.collide_radius, color=color, fill=False
            )
            plt.gca().add_patch(circle)

        # Plot streaming connections
        for agent_id, agent_position in self.agent_positions.items():
            if hasattr(self, "_last_actions") and agent_id in self._last_actions:
                stream_target = self._last_actions[agent_id]["send_stream"]
                if stream_target != self.num_friendlies:  # If not "no stream"
                    # Line from streaming agent to receiving agent
                    receiver_id = f"friendly_{stream_target}"
                    receiver_pos = self.agent_positions[receiver_id]
                    plt.plot(
                        [agent_position[0], receiver_pos[0]],
                        [agent_position[1], receiver_pos[1]],
                        "g--",
                        alpha=0.5,
                    )

                    # Line from receiving agent to tower
                    plt.plot(
                        [receiver_pos[0], 0], [receiver_pos[1], 0], "g--", alpha=0.5
                    )

        # Set plot limits
        most_extreme = max(
            np.abs(
                [
                    pos
                    for positions in self.agent_positions.values()
                    for pos in positions
                ]
                + [self.target_pos[0], self.target_pos[1]]
            )
        )
        most_extreme = max(40, most_extreme)
        plt.xlim(-most_extreme - 5, most_extreme + 5)
        plt.ylim(-most_extreme - 5, most_extreme + 5)

        # Set equal aspect ratio
        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(True)
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
