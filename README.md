# Robust Swarm

[![](videos/simulation_20250329220448.mp4)](videos/simulation_20250329220448.mp4)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.in
python src/train.py --help
python src/replay.py --help
```

If the gpu cannot be found on, you may need to reinstall torch with the right version of cuda for example if you have cuda 12.1 installed, you can run:

```bash
pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --user
```

## Actions and Observations

![](images/actions-movement.png)

## Environment

![](images/diagram.png)

## Reward Function

For each agent $i$ with position $p_i$, the reward $R_i$ is:

$$
\begin{equation}
R_i = \begin{cases}
e^{-\frac{|d_t - d_{opt}|}{5}} \cdot e^{-\frac{\|p_r - \frac{p_i}{2}\|}{5}} & \text{if streaming} \\
-1 & \text{if } \|p_i - p_j\| < 2r_c \text{ for any agent } j \neq i
\end{cases}
\end{equation}
$$

where:

- $d_t = \|p_i - p_t\|$ is the distance to target
- $d_{opt}$ is the optimal target distance
- $p_r$ is the receiver position
- $r_c$ is the collision radius
