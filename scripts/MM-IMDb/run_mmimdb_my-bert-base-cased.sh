#!/usr/bin/env bash

export TASK=mmimdb
export DATA_DIR=data/${TASK}/
export OUTPUT_NAME=output
export MODEL=bert
export MODEL_NAME=bert-base-cased

export TRAIN_BATCH_SIZE=1
export EVAL_BATCH_SIZE=1
export DEFAULT_BATCH_SIZE=8
export DEFAULT_SAVE_STEPS=50
export SAVE_STEPS=$((${DEFAULT_BATCH_SIZE}/${TRAIN_BATCH_SIZE}*${DEFAULT_SAVE_STEPS}/2))
export DEFAULT_MAX_SEQ_LENGTH=128
export MAX_SEQ_LENGTH=$((${DEFAULT_BATCH_SIZE}/${TRAIN_BATCH_SIZE}*${DEFAULT_MAX_SEQ_LENGTH}/2))

export STAGE_NUM=1
export NEXT_STAGE_NUM=2

python ./examples/mm-imdb-my/run_mmimdb.py \
    --model_type ${MODEL} \
    --model_name_or_path ${MODEL_NAME} \
    --image_model ./models/resnet/resnet152.pth \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_NAME}/${TASK}/${TASK}-${MODEL_NAME}/stage_${NEXT_STAGE_NUM} \
    --max_seq_len ${DEFAULT_MAX_SEQ_LENGTH} \
    --per_gpu_train_batch_size ${TRAIN_BATCH_SIZE}   \
    --per_gpu_eval_batch_size ${EVAL_BATCH_SIZE}   \
    --save_steps ${DEFAULT_SAVE_STEPS} \
    --num_image_embeds 3 \
    --num_train_epochs 1 \
    --eval_all_checkpoints  \
    --do_eval  \
    --do_train \
#    --do_test  \
