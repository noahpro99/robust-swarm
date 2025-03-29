# CAAS

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.in
python src/train.py --help
python src/replay.py --help
python src/test.py --help
```

If the gpu cannot be found on Jupiter Hub, you may need to reinstall torch

```bash
pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --user
```

## Viewing Replays in Jupiter Hub

You can use the following to view a replay in Jupiter Hub in a python cell. Better not to commit the notebook with the video in it however.

```python
from IPython.display import Video
path = "../" + "episodes/simulation_20240727090801.mp4"
Video(path, embed=True, width=500)
```

# Results Log

## 2024-06-10 Policy

- penalty for being too close to each other

![](./results/old_episodes/simulation_20240610164352.mp4)
<video src="./results/old_episodes/simulation_20240610164352.mp4" controls preload></video>

## 2024-06-13 Policy

- prey starts closer to the agents
- prey slightly slower than agents
- penalty for being close to other agents once they are close to the prey instead of all the time
- penalty for being too close to the prey
- relative distance observation
- max view distance observation

![](./results/old_episodes/simulation_20240610210803.mp4)
<video src="./results/old_episodes/simulation_20240610210803.mp4" controls preload></video>

## 2024-06-17 Policy

- observation of the prey is last seen location updated every x steps
- observation of the prey last seen location has a noise

![](./results/old_episodes/simulation_20240617140154.mp4)
<video src="./results/old_episodes/simulation_20240617140154.mp4" controls preload></video>

![](./results/old_episodes/simulation_20240617142931.mp4)
<video src="./results/old_episodes/simulation_20240617142931.mp4" controls preload></video>

![Test Results](results/trial_results_6_17/rewards_6_17.png)

## 2024-06-21 Policy

- slower prey and less random movement
- last seen location is picked with the noise instead
- reward is now
  - distance to min distance to prey
  - maximize distance to other agents
  - minimize distance from mean of agents to prey
  - minimize distance to the mean of the agents
- test script that evaluates changes to the env vs reward

![](./results/old_episodes/simulation_20240621064109.mp4)
<video src="./results/old_episodes/simulation_20240621064109.mp4" controls preload></video>

![Test Results](results/trial_results_6_21/rewards_6_21.png)

## 2024-06-26 Policy

- observation
  - relative distance to last seen location
  - **relative distance to previous last seen location**
  - relative distance to closest agent within view distance
  - **number of steps since last seen location update**
- reward
  - minimize distance to the optimal distance to prey
  - maximize distance to other agents
  - minimize distance from mean of agents to prey
  - minimize distance to the mean of the agents
- **debugging histograms and test plot is split by reward type**

![Test Results](results/trial_results_6_26/results.png)

## 2024-07-01 Policy

- observation
  - list of entities relative position and type
- reward
  - penalize by step count for not capturing every step
  - penalty for crashing into any entity either agent or target
  - reward for capturing the target
  - slight guiding reward for being close to the target
  - slight guiding reward for maximizing mean distance to other agents
- prey behaviour
  - moves away from nearest agent in its view distance
  - can't move in a direction that would point into an agent's scare arms
  - finds closest direction to the one that is the furthest from the nearest agent
- done_condition
  - the prey is not able to pick any direction to move because all directions point to a position that points into an agent's scare arms

![](./results/old_episodes/simulation_20240701181929.mp4)
<video src="./results/old_episodes/simulation_20240701181929.mp4" controls preload></video>

## 2024-07-04

- Observation
  - relative estimated position (noise in direction and distance based on distance)

![Observation distribution](./images/observationDistributionImage.png)

- Experimented to remove part of the reward causing gradient explosion
- New testing with new plots

![](./results/old_episodes/simulation_20240703034007.mp4)
<video src="./results/old_episodes/simulation_20240703034007.mp4" controls preload></video>

![](./results/old_episodes/simulation_20240703035850.mp4)
<video src="./results/old_episodes/simulation_20240703035850.mp4" controls preload></video>

![Noise Direction Trial Results](./results/trial_results_7_04/noise_direction_analysis.png)

## 2024-07-10

- Observation for each other entity only updates every x steps where x is from a normal distribution
- Observation of how many steps since last update for each entity

![](./results/trial_results_7_08/simulation_20240708135709.mp4)
<video src="./results/trial_results_7_08/simulation_20240708135709.mp4" controls preload></video>

![](./results/trial_results_7_08/simulation_20240708141517.mp4)
<video src="./results/trial_results_7_08/simulation_20240708141517.mp4" controls preload></video>

![Noise Direction Trial Results](./results/trial_results_7_08/noise_direction_analysis.png)

## 2024-07-31

- Prey is controlled by policy
- Observation
  - relative estimated position (noise in direction and distance based on distance)
  - previous relative estimated position
  - entity type
  - number of steps since last update for each entity
  - absolute position of self
- Reward
  - penalize for collision with any entity
  - prey rewarded for maximizing number of steps of episode
  - Agents
    - rewarded for minimizing number of steps of episode
    - guiding reward for being close to the prey and further from each other

![](./results/trial_results_7_31/simulation_20240730164503.mp4)
<video src="./results/trial_results_7_31/simulation_20240730164503.mp4" controls preload></video>

![](./results/trial_results_7_31/simulation_20240730164542.mp4)
<video src="./results/trial_results_7_31/simulation_20240730164542.mp4" controls preload></video>

![Steps analysis](./results/trial_results_7_31/Steps_analysis.png)
