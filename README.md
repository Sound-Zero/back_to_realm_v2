# Back To Realm v2

Back To Realm v2 is a local reinforcement-learning project for training a PPO agent in a treasure-hunt style grid environment. It includes:

- a deterministic local environment (`env.Env_v2`)
- feature preprocessing and PPO sample packing
- actor and learner network modules built with PyTorch
- single-process and multi-process training entry points
- a Streamlit dashboard for training metrics

## Repository Status

This repository has been refactored toward GitHub open-source project conventions. Historical generated files such as checkpoints, logs, Python caches, and local metrics are ignored by Git and should not be committed.

## Quick Start

Create an environment with Python 3.10 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dashboard,dev]"
```

Run tests:

```powershell
pytest
```

Run single-process training:

```powershell
python -m PPO.workflow.train_workflow
```

Run multi-actor training:

```powershell
python launcher.py --actors 4 --device cuda
```

Run the dashboard:

```powershell
streamlit run dashboard.py
```

## Project Layout

```text
PPO/
  agent.py                 Agent wrapper for preprocessing, action sampling, and learning
  algorithm/algorithm.py   PPO learner and loss computation
  conf/conf.py             Training and feature dimensions
  feature/                 Feature preprocessing and sample packing
  model/model.py           Actor and learner neural networks
  workflow/                Training rollout workflow
env/
  env_v2.py                Local treasure-hunt environment
  config.toml              Environment configuration
map/                       Map assets used by the local environment
dashboard.py               Streamlit metrics dashboard
launcher.py                Multi-process actor/learner launcher
tests/                     Pytest coverage for core contracts
```

## Notes For Contributors

- Keep checkpoints, logs, metrics, and caches out of Git.
- Preserve the feature/sample dimension contracts unless tests and documentation are updated together.
- Prefer small, focused pull requests with tests for behavioral changes.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
