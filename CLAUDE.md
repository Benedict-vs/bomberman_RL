@AGENTS.md

## Claude Code specifics

- `AGENTS.md` above is the shared, tool-agnostic project spec — edit that file, not this one,
  for anything a collaborator's agent should also know.
- Before touching `agent_code/<name>/callbacks.py` or `train.py`, check the framework contract in
  `AGENTS.md` — `act()` has a 0.5 s budget and must return one of the six action strings.
- Don't "helpfully" refactor the provided framework files (`environment.py`, `agents.py`,
  `items.py`, `settings.py`, `main.py`). They are reset to upstream for the tournament; changes
  there are training-only scaffolding and must be called out explicitly.
- Prefer `uv run python main.py ...` over bare `python` so the project venv is used.
- At the start of a session, skim the logbook of whoever you're working with (`BENEDICT.md`,
  `MAXI.md`, `BEN.md`) for the last state and the *why* behind past decisions — the git history
  only shows the *what*. Append an entry when asked; don't touch someone else's file.
