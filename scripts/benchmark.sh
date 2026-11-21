#!/usr/bin/env bash
# The benchmark: the other model variants, trained like the main ones, then everything tested on the same data.
# Run after scripts/train_all.sh (it reuses data/ and writes to the same checkpoints/ and docs/).
#
#   bash scripts/benchmark.sh              # a few hours
#   QUICK=1 bash scripts/benchmark.sh      # 20 steps per stage in /tmp: checks it all works
#
#   rhoa_small, rhoa_small_scratch   2 layers x 64 wide, with and without pre-training
#   rhoa_gpt2_scratch                4 x 128 with GPT-2 blocks (LayerNorm, learned positions, GELU), no pre-training
#   melon_small                      Mēlon, 2 layers x 64 wide
#   melon_maze                       Mēlon learns the maze game from pictures (+ its demo video)
# then evaluate_all.py tests every checkpoint (+ baselines) and plot_metrics.py draws the benchmark chart.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PYTHONUNBUFFERED=1

SMALL="--n_layer 2 --n_embd 64 --n_head 4"
GPT2="--n_layer 4 --n_embd 128 --n_head 4 --norm_type layernorm --pos_encoding learned --gated_mlp false --activation gelu_tanh --bias true"
TASKS="babi,clinc150,banking77,csqa"
if [ "${QUICK:-0}" = "1" ]; then
    OUT=/tmp/tiny-dlms-quick; CKPT=$OUT/checkpoints; LOGS=$OUT/logs; SITE=$OUT/docs
    MELON_STEPS=20; MAZE_STEPS=20; PRETRAIN_STEPS=20; FINETUNE_STEPS=20; EVAL_EVERY=10
    TEST_LIMIT="--max_decisions 50 --mazes 30"; TRACKS=1
else
    CKPT=checkpoints; LOGS=logs; SITE=docs
    MELON_STEPS=4000; MAZE_STEPS=6000; PRETRAIN_STEPS=6000; FINETUNE_STEPS=16000; EVAL_EVERY=500
    TEST_LIMIT=""; TRACKS=5
fi
mkdir -p "$CKPT" "$LOGS" "$SITE/assets"
steps() { echo "--max_iters $1 --lr_decay_iters $1 --eval_interval $EVAL_EVERY --out_dir $CKPT"; }
stage() { echo; echo "== $(date +%H:%M) $*"; }
done_() { [ -f "$CKPT/$1" ] && echo "   $1 already trained: kept (delete it to train again)"; }

stage "Mēlon plays the maze (4 x 128)"
$PY scripts/maze_sanity_check.py --n_layer 4 --n_embd 128 --picture "$CKPT/maze_batch.png"
done_ melon_maze.pt || $PY scripts/maze_train.py --n_layer 4 --n_embd 128 --n_head 4 $(steps $MAZE_STEPS) --ckpt_name melon_maze.pt \
    > "$LOGS/melon_maze.log" 2>&1
$PY scripts/maze_play.py --ckpt "$CKPT/melon_maze.pt" --mazes 50 --video "$SITE/assets/maze.mp4" "$SITE/assets/maze.gif"

stage "Mēlon small (2 x 64)"
done_ melon_small.pt || $PY scripts/car_train.py $SMALL $(steps $MELON_STEPS) --ckpt_name melon_small.pt > "$LOGS/melon_small.log" 2>&1

stage "Rhoa small (2 x 64): pre-training, then fine-tuning"
done_ rhoa_small_pretrain.pt || $PY scripts/train.py --data_dir data/pretrain $SMALL --drop_rate 0 $(steps $PRETRAIN_STEPS) \
    --ckpt_name rhoa_small_pretrain.pt > "$LOGS/rhoa_small_pretrain.log" 2>&1
done_ rhoa_small.pt || $PY scripts/finetune.py --init_from "$CKPT/rhoa_small_pretrain.pt" --tasks $TASKS $(steps $FINETUNE_STEPS) \
    --ckpt_name rhoa_small.pt > "$LOGS/rhoa_small.log" 2>&1

stage "Rhoa small without pre-training"
done_ rhoa_small_scratch.pt || $PY scripts/finetune.py --init_from scratch $SMALL --drop_rate 0 --tasks $TASKS $(steps $FINETUNE_STEPS) \
    --ckpt_name rhoa_small_scratch.pt > "$LOGS/rhoa_small_scratch.log" 2>&1

stage "Rhoa with GPT-2 blocks, without pre-training"
done_ rhoa_gpt2_scratch.pt || $PY scripts/finetune.py --init_from scratch $GPT2 --drop_rate 0 --tasks $TASKS $(steps $FINETUNE_STEPS) \
    --ckpt_name rhoa_gpt2_scratch.pt > "$LOGS/rhoa_gpt2_scratch.log" 2>&1

stage "test every model on the same data"
$PY scripts/evaluate_all.py --ckpt_dir "$CKPT" --tracks $TRACKS $TEST_LIMIT --out "$SITE/assets/results.json"
$PY scripts/plot_metrics.py --ckpt_dir "$CKPT" --out "$SITE/assets/plots"
stage "done"
