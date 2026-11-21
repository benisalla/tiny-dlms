#!/usr/bin/env bash
# Train, test and record every model of the project, one after the other.
#
#   bash scripts/train_all.sh              # the real run (a few hours): models in checkpoints/, logs in logs/,
#                                          # results, graphs, videos and the web demo in docs/
#   QUICK=1 bash scripts/train_all.sh      # the same pipeline with a few steps per stage, written to /tmp:
#                                          # proves everything works end to end before the long run
#
#   1. health checks     sanity_check.py, car_sanity_check.py (stop if one fails)
#   2. Mēlon  (images)   car_train.py: behaviour cloning on CarRacing frames
#   3. Rhoa   (text)     train.py (pre-training) -> finetune.py on babi, clinc150, banking77, csqa
#   4. Rhoa without pre-training: the same fine-tuning from random weights (the comparison)
#   5. final test        evaluate_all.py (held-out test sets, unseen tracks) -> results.json
#   6. records           plot_metrics.py (graphs), car_drive.py + rhoa_demo.py (videos), export_web.py (web demo)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PYTHONUNBUFFERED=1  # logs show progress live

SIZE="--n_layer 4 --n_embd 128 --n_head 4"
TASKS="babi,clinc150,banking77,csqa"
if [ "${QUICK:-0}" = "1" ]; then
    OUT=/tmp/tiny-dlms-quick; CKPT=$OUT/checkpoints; LOGS=$OUT/logs; SITE=$OUT/docs
    MELON_STEPS=20; PRETRAIN_STEPS=20; FINETUNE_STEPS=20; EVAL_EVERY=10; TEST_LIMIT="--max_decisions 50"; TRACKS=1
else
    CKPT=checkpoints; LOGS=logs; SITE=docs
    MELON_STEPS=4000; PRETRAIN_STEPS=6000; FINETUNE_STEPS=16000; EVAL_EVERY=500; TEST_LIMIT=""; TRACKS=5
fi
mkdir -p "$CKPT" "$LOGS" "$SITE/assets"
steps() { echo "--max_iters $1 --lr_decay_iters $1 --eval_interval $EVAL_EVERY --out_dir $CKPT"; }
stage() { echo; echo "== $(date +%H:%M) $*"; }
done_() { [ -f "$CKPT/$1" ] && echo "   $1 already trained: kept (delete it to train again)"; }

stage "data"
[ -f data/clinc150/train.csv ] || $PY scripts/prepare_data.py
[ -f data/car/train.pt ] || $PY scripts/car_record.py

stage "1. health checks"
$PY scripts/sanity_check.py
$PY scripts/car_sanity_check.py $SIZE --picture "$CKPT/car_batch.png"

stage "2. Mēlon (images)"
done_ melon.pt || $PY scripts/car_train.py $SIZE $(steps $MELON_STEPS) --ckpt_name melon.pt > "$LOGS/melon.log" 2>&1

stage "3. Rhoa: pre-training, then fine-tuning"
done_ rhoa_pretrain.pt || $PY scripts/train.py --data_dir data/pretrain $SIZE --drop_rate 0 $(steps $PRETRAIN_STEPS) \
    --ckpt_name rhoa_pretrain.pt > "$LOGS/rhoa_pretrain.log" 2>&1
done_ rhoa.pt || $PY scripts/finetune.py --init_from "$CKPT/rhoa_pretrain.pt" --tasks $TASKS $(steps $FINETUNE_STEPS) \
    --ckpt_name rhoa.pt > "$LOGS/rhoa.log" 2>&1

stage "4. Rhoa without pre-training (the comparison)"
done_ rhoa_scratch.pt || $PY scripts/finetune.py --init_from scratch $SIZE --drop_rate 0 --tasks $TASKS $(steps $FINETUNE_STEPS) \
    --ckpt_name rhoa_scratch.pt > "$LOGS/rhoa_scratch.log" 2>&1

stage "5. final test on held-out data"
$PY scripts/evaluate_all.py --ckpt_dir "$CKPT" --tracks $TRACKS $TEST_LIMIT --out "$SITE/assets/results.json"

stage "6. graphs, videos, web demo"
$PY scripts/plot_metrics.py --ckpt_dir "$CKPT" --out "$SITE/assets/plots"
$PY scripts/car_drive.py --ckpt "$CKPT/melon.pt" --drivers model --tracks 1 \
    --video "$SITE/assets/melon.mp4" "$SITE/assets/melon.gif"
$PY scripts/rhoa_demo.py --ckpt "$CKPT/rhoa.pt" --out "$SITE/assets/rhoa.mp4" "$SITE/assets/rhoa.gif"
$PY scripts/export_web.py --ckpt "$CKPT/rhoa.pt" --out_dir "$SITE/demo"

stage "done"
grep -h -E "^done|accuracy" "$LOGS"/*.log || true
