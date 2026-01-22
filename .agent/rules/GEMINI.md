---
trigger: always_on
---

Interact and run the code only via `hare run --rm -it -v .:/app -e WANDB_API_KEY=b5c158bd2412bc120b3dfb33570d67747f4fdac2 --gpus device=0 $USER/turbozero `. Only use gpu 0.

`hare` is the alias for docker. Don't leave dangling containers (fzn21/turbozero) that fill GPU memory, use `hare ps` and `hare kill `.

Validate via `bash scripts/validate--train-eval-wandb.sh`, directly update its specified args if needed.

`uv` is the package manager.

My goal is to (i) reproduce AlphaZero one-to-one, and (ii) run FastSVERL in parallel, capturing all the same metrics the paper did.

FastSVERL paper: `./FastSVERL.pdf`
FastSVERL public code: https://github.com/djeb20/fastsverl
FASTSVERL public code's Shapley network: https://github.com/djeb20/fastsverl/blob/main/fastsverl/fastsverl/shapley.py
