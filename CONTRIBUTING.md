# Contributing

Thank you for improving Back To Realm v2.

## Development Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dashboard,dev]"
```

## Pull Request Checklist

- Add or update tests for behavior changes.
- Run `pytest` before opening the pull request.
- Run `ruff check .` when changing Python code.
- Do not commit generated files such as checkpoints, logs, caches, or local metrics.
- Document changes that affect training commands, feature dimensions, or environment configuration.

## Coding Guidelines

- Keep feature and sample dimensions aligned with `PPO.conf.conf.Config`.
- Keep training output paths configurable or ignored by Git.
- Prefer clear logging over print statements in library code.
- Avoid broad rewrites unless they simplify a specific tested behavior.
