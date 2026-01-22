#!/bin/bash

# This script runs a short training session to verify that the training, 
# evaluation, and wandb logging are working correctly on the current system.
# It uses Go 9x9, a small batch size, and a small buffer to avoid OOM.

echo "Starting short validation run with Go 9x9 and small batch size..."

hare run --rm -it -v .:/app -e WANDB_API_KEY=b5c158bd2412bc120b3dfb33570d67747f4fdac2 --gpus device=0 fzn21/turbozero uv run src/train.py \
    --env-id go_9x9 \
    --pred-shapley-weight 0.05 \
    --bhvr-char-weight 0.05 \
    --bhvr-shapley-weight 0.05 \
    --shapley-update-ratio 1 \
    --num-epochs 1 \
    --batch-size 128 \
    --train-batch-size 128 \
    --collection-steps-per-epoch 8 \
    --train-steps-per-epoch 8 \
    --warmup-steps 0 \
    --eval-num-iterations 10 \
    --eval-test-num-iterations 10 \
    --tester-num-episodes 4 \
    --elo-eval-num-episodes 4 \
    --eval_every 1 \
    --eval-greedy \
    --eval-selfplay \
    --buffer-capacity 1000
