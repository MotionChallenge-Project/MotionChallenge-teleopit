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

[Provenance](provenance.json)

MuJoCo 3.17 batch is the default. Use `--jobs 8` for parallel clips;
download and evaluation progress updates every 5 seconds.


## Checkpoint source

[BotRunner64/Teleopit](https://github.com/BotRunner64/Teleopit/blob/f9263865c581802ad531854b8e547e2403a945f3/README.md) · [track_g1.onnx](https://huggingface.co/12e21/Teleopit-models/blob/94cf996444fea6894b87c28e86606cd4c2f1408f/checkpoints/track_g1.onnx). Author-released checkpoint.

The adapter pins this published artifact by SHA256. Artifact identity is separate from inference parity.

One `uv run evaluate.py` produces one verified `result.json`: success, completion,
three MPBPE alignments through first failure and the same three over the full motion.
Both scopes use the same checkpoint and source revision; no branch switching is needed.
