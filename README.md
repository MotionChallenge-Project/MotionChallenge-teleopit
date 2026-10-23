# TeleopIT G1

Independent Motion Challenge adapter for [TeleopIT G1](https://github.com/BotRunner64/Teleopit).

```sh
git clone --recurse-submodules https://github.com/motion-challenge-project/motion-challenge-teleopit.git
cd motion-challenge-teleopit
uv sync
uv run evaluate.py
```

The root `motion-challenge` submodule owns datasets, physics, failure rules and scoring.
Local weights are reused; downloads are hash-checked. Results are stored outside Git.
Use `--weights PATH --data PATH --offline` for cached artifacts.

[Provenance](provenance.json) · [Website](https://motion-challenge-project.github.io/motion-challenge-project/)

MuJoCo 3.17 batch is the default. Use `--jobs 8` for parallel clips;
download and evaluation progress updates every 5 seconds.

Run all frames after tracking failure: `uv run evaluate.py --termination none`.
This records full-motion errors under a separate protocol; success still requires no failure.
