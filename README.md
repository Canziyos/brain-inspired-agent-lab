# Brain Lab

A small brain-inspired agent laboratory. It currently combines a symbolic GridWorld, a Gymnasium-compatible environment backend, rule-based motivation and planning, and a small neural reward predictor.

The project is a practical sandbox for building toward embodied cognitive-agent experiments.

## Current Capabilities

- Deterministic GridWorld generation with food, danger, and mystery objects.
- A Gymnasium environment wrapper with `reset()` and `step()`.
- Optional Pygame rendering for the GridWorld backend.
- Rule-based action evaluation, goal selection, and frontier planning.
- A PyTorch `ImmediateRewardNetwork` that learns immediate observed reward.
- Run logging and CSV step-metric export.
- Behavioural tests for dynamics, planning, environment termination, output writing, and seeded runs.

## Architecture

```text
src/core         actions, agent state, dynamics, perception, world
src/envs         Gymnasium environment backend
src/learning     features, rewards, samples, neural reward network
src/planning     goal and frontier planning
src/policies     rule policy and neural shadow policy
src/simulation   setup, step loop, runner, metrics
src/diagnostics  run output and logging
src/views        animation and plots
tests            behavioural safeguards
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## How To Run

Run the default simulation:

```powershell
python main.py
```

Run the Gymnasium/Pygame GridWorld demo:

```powershell
python experiments\gym_gridworld_demo.py
```

Run a headless simulation from Python:

```python
from src.config import SimulationConfig
from src.simulation.runner import run_simulation

history = run_simulation(
    SimulationConfig(
        verbose=False,
        show_animation=False,
        show_plots=False,
    )
)
```

## How To Test

```powershell
python -m pytest -q
```

For a syntax pass:

```powershell
python -m compileall src tests experiments main.py
```

## Neural Versus Hand-Coded

Neural:

- `ImmediateRewardNetwork`, a small feed-forward PyTorch model with learnable weights.
- The network predicts immediate scalar reward for candidate state-action features.

Hand-coded:

- World generation and dynamics.
- Rule-policy scoring and motivation.
- Goal and frontier planning.
- Reward calculation.
- Gymnasium environment wrapping and rendering.

## Current Limitations

- The neural model predicts immediate reward, not long-term value.
- The rule policy still controls behaviour; the neural policy is observed in shadow.
- The GridWorld is symbolic and discrete.
- The Gymnasium backend still exposes some internal state to the existing controller bridge.
- There is no ROS 2, Gazebo, MuJoCo, spiking neural model, or biologically detailed plasticity yet.

