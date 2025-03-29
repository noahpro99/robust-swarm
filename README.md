# Robust Swarm

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
