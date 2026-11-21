#!/usr/bin/env bash
# Real pre-training: Rhoa reads ~1.5 GB of web text (scripts/prepare_web.py), then learns the 4 decision tasks.
# The comparison, all 4 x 128 with the same web BPE tokenizer, so only the pre-training differs:
#
#   rhoa_web_pretrain -> rhoa_web     pre-trained on the web corpus, then fine-tuned
#   rhoa_web_scratch                  no pre-training (same tokenizer), fine-tuned the same way
#
# (rhoa / rhoa_scratch, from scripts/train_all.sh, are the same model pre-trained on the small in-domain corpus.)
#
#   bash scripts/web_pretrain.sh            # pre-training ~1-2 hours, then ~1 hour of fine-tuning
#   QUICK=1 bash scripts/web_pretrain.sh    # 20 steps per stage in /tmp: checks it all works
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PYTHONUNBUFFERED=1

SIZE="--n_layer 4 --n_embd 128 --n_head 4 --block_size ${BLOCK:-128}"
TASKS="babi,clinc150,banking77,csqa"
WEB=data/web
if [ "${QUICK:-0}" = "1" ]; then
    OUT=/tmp/tiny-dlms-quick; CKPT=$OUT/checkpoints; LOGS=$OUT/logs; SITE=$OUT/docs; WEB=${WEB_DIR:-$WEB}
    PRETRAIN_STEPS=20; FINETUNE_STEPS=20; EVAL_EVERY=10; PRE_EVAL=10; TEST_LIMIT="--max_decisions 50 --mazes 30"; TRACKS=1
else
    CKPT=checkpoints; LOGS=logs; SITE=docs
    PRETRAIN_STEPS=${PRETRAIN_STEPS:-30000}; FINETUNE_STEPS=16000; EVAL_EVERY=500; PRE_EVAL=1000; TEST_LIMIT=""; TRACKS=5
fi
mkdir -p "$CKPT" "$LOGS" "$SITE/assets"
stage() { echo; echo "== $(date +%H:%M) $*"; }
done_() { [ -f "$CKPT/$1" ] && echo "   $1 already trained: kept (delete it to train again)"; }

stage "data"
[ -f $WEB/train.bin ] || $PY scripts/prepare_web.py --out $WEB

stage "health checks on the web corpus"
$PY scripts/sanity_check.py --web_dir $WEB --block_size ${BLOCK:-128}

stage "pre-training on the web corpus"
# 128 windows of 127 tokens per step (~16k tokens); 30k steps read ~490M tokens, about one pass over the corpus
done_ rhoa_web_pretrain.pt || $PY scripts/train.py --data_dir $WEB $SIZE --drop_rate 0 --batch_size 128 \
    --max_iters $PRETRAIN_STEPS --lr_decay_iters $PRETRAIN_STEPS --warmup_iters 1000 --eval_interval $PRE_EVAL \
    --out_dir $CKPT --ckpt_name rhoa_web_pretrain.pt > "$LOGS/rhoa_web_pretrain.log" 2>&1

stage "fine-tuning: pre-trained on the web, and from scratch with the same tokenizer"
steps="--max_iters $FINETUNE_STEPS --lr_decay_iters $FINETUNE_STEPS --eval_interval $EVAL_EVERY --out_dir $CKPT"
done_ rhoa_web.pt || $PY scripts/finetune.py --init_from "$CKPT/rhoa_web_pretrain.pt" --tasks $TASKS $steps \
    --ckpt_name rhoa_web.pt > "$LOGS/rhoa_web.log" 2>&1
done_ rhoa_web_scratch.pt || $PY scripts/finetune.py --init_from scratch $SIZE --drop_rate 0 \
    --tokenizer_path $WEB/tokenizer.json --tasks $TASKS $steps --ckpt_name rhoa_web_scratch.pt > "$LOGS/rhoa_web_scratch.log" 2>&1

stage "test every model on the same data"
$PY scripts/evaluate_all.py --ckpt_dir "$CKPT" --tracks $TRACKS $TEST_LIMIT --out "$SITE/assets/results.json"
$PY scripts/plot_metrics.py --ckpt_dir "$CKPT" --out "$SITE/assets/plots"
stage "done"
