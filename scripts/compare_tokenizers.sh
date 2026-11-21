#!/usr/bin/env bash
# Word tokenizer vs BPE: the same Rhoa models, trained the same way, only the tokenizer changes.
# The word-tokenizer models come from scripts/train_all.sh and scripts/benchmark.sh (rhoa, rhoa_scratch, rhoa_small...).
#
#   bash scripts/compare_tokenizers.sh            # about 1.5 hours
#   QUICK=1 bash scripts/compare_tokenizers.sh    # 20 steps per stage in /tmp: checks it all works
#
#   rhoa_bpe_pretrain -> rhoa_bpe           4 x 128, BPE, pre-trained   (vs rhoa)
#   rhoa_bpe_scratch                        4 x 128, BPE, no pre-training (vs rhoa_scratch)
#   rhoa_small_bpe_pretrain -> rhoa_small_bpe, rhoa_small_bpe_scratch   2 x 64 (vs rhoa_small, rhoa_small_scratch)
# then evaluate_all.py tests every checkpoint and plot_metrics.py draws the charts.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PYTHONUNBUFFERED=1

BIG="--n_layer 4 --n_embd 128 --n_head 4"
SMALL="--n_layer 2 --n_embd 64 --n_head 4"
TOK="--tokenizer bpe --bpe_vocab_size 4096"
TASKS="babi,clinc150,banking77,csqa"
if [ "${QUICK:-0}" = "1" ]; then
    OUT=/tmp/tiny-dlms-quick; CKPT=$OUT/checkpoints; LOGS=$OUT/logs; SITE=$OUT/docs
    PRETRAIN_STEPS=20; FINETUNE_STEPS=20; EVAL_EVERY=10; TEST_LIMIT="--max_decisions 50 --mazes 30"; TRACKS=1
else
    CKPT=checkpoints; LOGS=logs; SITE=docs
    PRETRAIN_STEPS=6000; FINETUNE_STEPS=16000; EVAL_EVERY=500; TEST_LIMIT=""; TRACKS=5
fi
mkdir -p "$CKPT" "$LOGS" "$SITE/assets"
steps() { echo "--max_iters $1 --lr_decay_iters $1 --eval_interval $EVAL_EVERY --out_dir $CKPT"; }
stage() { echo; echo "== $(date +%H:%M) $*"; }
done_() { [ -f "$CKPT/$1" ] && echo "   $1 already trained: kept (delete it to train again)"; }

stage "health checks with BPE"
$PY scripts/sanity_check.py --tokenizer bpe

for size in big small; do
    if [ $size = big ]; then CFG=$BIG; NAME=rhoa; else CFG=$SMALL; NAME=rhoa_small; fi
    stage "$NAME with BPE: pre-training, then fine-tuning"
    done_ ${NAME}_bpe_pretrain.pt || $PY scripts/train.py --data_dir data/pretrain $CFG $TOK --drop_rate 0 \
        $(steps $PRETRAIN_STEPS) --ckpt_name ${NAME}_bpe_pretrain.pt > "$LOGS/${NAME}_bpe_pretrain.log" 2>&1
    done_ ${NAME}_bpe.pt || $PY scripts/finetune.py --init_from "$CKPT/${NAME}_bpe_pretrain.pt" --tasks $TASKS \
        $(steps $FINETUNE_STEPS) --ckpt_name ${NAME}_bpe.pt > "$LOGS/${NAME}_bpe.log" 2>&1
    stage "$NAME with BPE, without pre-training"
    done_ ${NAME}_bpe_scratch.pt || $PY scripts/finetune.py --init_from scratch $CFG $TOK --drop_rate 0 --tasks $TASKS \
        $(steps $FINETUNE_STEPS) --ckpt_name ${NAME}_bpe_scratch.pt > "$LOGS/${NAME}_bpe_scratch.log" 2>&1
done

stage "test every model on the same data"
$PY scripts/evaluate_all.py --ckpt_dir "$CKPT" --tracks $TRACKS $TEST_LIMIT --out "$SITE/assets/results.json"
$PY scripts/plot_metrics.py --ckpt_dir "$CKPT" --out "$SITE/assets/plots"
stage "done"
