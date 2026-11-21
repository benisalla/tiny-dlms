#!/usr/bin/env bash
# Round 2 of real pre-training: every Rhoa version reads 2 GB of web text (data/web2), then learns the 4 tasks.
# data/web2 uses the same tokenizer as data/web (round 1, scripts/web_pretrain.sh), so the two rounds differ only by
# the amount of text, and every model has a twin without pre-training that uses that same tokenizer.
#
#   rhoa_web2        4 x 128                 (twin: rhoa_web_scratch, from round 1)
#   rhoa_small_web2  2 x 64                  (twin: rhoa_small_web_scratch)
#   rhoa_gpt2_web2   4 x 128, GPT-2 blocks   (twin: rhoa_gpt2_web_scratch); lr 3e-4, which suits GPT-2 blocks
#
#   bash scripts/web_pretrain_all.sh            # ~8 hours
#   QUICK=1 bash scripts/web_pretrain_all.sh    # 20 steps per stage in /tmp: checks it all works
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PYTHONUNBUFFERED=1

TASKS="babi,clinc150,banking77,csqa"
WEB=data/web2
BIG="--n_layer 4 --n_embd 128 --n_head 4"
SMALL="--n_layer 2 --n_embd 64 --n_head 4"
GPT2="--n_layer 4 --n_embd 128 --n_head 4 --norm_type layernorm --pos_encoding learned --gated_mlp false --activation gelu_tanh --bias true"
if [ "${QUICK:-0}" = "1" ]; then
    OUT=/tmp/tiny-dlms-quick; CKPT=$OUT/checkpoints; LOGS=$OUT/logs; SITE=$OUT/docs
    PRETRAIN_STEPS=20; FINETUNE_STEPS=20; EVAL_EVERY=10; PRE_EVAL=10; TEST_LIMIT="--max_decisions 50 --mazes 30"; TRACKS=1
else
    CKPT=checkpoints; LOGS=logs; SITE=docs
    # 40k steps x 128 windows x 127 tokens = 650M tokens, ~1.36 passes over 2 GB (the same number of passes as round 1)
    PRETRAIN_STEPS=${PRETRAIN_STEPS:-40000}; FINETUNE_STEPS=16000; EVAL_EVERY=500; PRE_EVAL=1000; TEST_LIMIT=""; TRACKS=5
fi
mkdir -p "$CKPT" "$LOGS" "$SITE/assets"
stage() { echo; echo "== $(date +%H:%M) $*"; }
done_() { [ -f "$CKPT/$1" ] && echo "   $1 already trained: kept (delete it to train again)"; }
fine="--max_iters $FINETUNE_STEPS --lr_decay_iters $FINETUNE_STEPS --eval_interval $EVAL_EVERY --out_dir $CKPT"

stage "data"
[ -f $WEB/train.bin ] || $PY scripts/prepare_web.py --out $WEB --max_gb 2 --tokenizer data/web/tokenizer.json

stage "health checks on the 2 GB corpus"
$PY scripts/sanity_check.py --web_dir $WEB

# name | size | learning rate (pre-training and fine-tuning) | twin without pre-training
for run in "rhoa_web2|$BIG|1e-3|rhoa_web_scratch" "rhoa_small_web2|$SMALL|1e-3|rhoa_small_web_scratch" \
           "rhoa_gpt2_web2|$GPT2|3e-4|rhoa_gpt2_web_scratch"; do
    IFS="|" read -r NAME CFG LR TWIN <<< "$run"
    LRS="--learning_rate $LR --min_lr $(python -c "print($LR / 10)")"
    stage "$NAME: pre-training on 2 GB of web text"
    done_ ${NAME}_pretrain.pt || $PY scripts/train.py --data_dir $WEB $CFG $LRS --drop_rate 0 --batch_size 128 \
        --max_iters $PRETRAIN_STEPS --lr_decay_iters $PRETRAIN_STEPS --warmup_iters 1000 --eval_interval $PRE_EVAL \
        --out_dir $CKPT --ckpt_name ${NAME}_pretrain.pt > "$LOGS/${NAME}_pretrain.log" 2>&1
    stage "$NAME: fine-tuning"
    done_ $NAME.pt || $PY scripts/finetune.py --init_from "$CKPT/${NAME}_pretrain.pt" $LRS --tasks $TASKS $fine \
        --ckpt_name $NAME.pt > "$LOGS/$NAME.log" 2>&1
    stage "$TWIN: the same model without pre-training (same tokenizer)"
    done_ $TWIN.pt || $PY scripts/finetune.py --init_from scratch $CFG $LRS --drop_rate 0 \
        --tokenizer_path $WEB/tokenizer.json --tasks $TASKS $fine --ckpt_name $TWIN.pt > "$LOGS/$TWIN.log" 2>&1
done

stage "test every model on the same data"
$PY scripts/evaluate_all.py --ckpt_dir "$CKPT" --tracks $TRACKS $TEST_LIMIT --out "$SITE/assets/results.json"
$PY scripts/plot_metrics.py --ckpt_dir "$CKPT" --out "$SITE/assets/plots"
stage "done"
