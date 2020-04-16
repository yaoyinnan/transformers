#!/usr/bin/env bash
set -eux

export TASK=FakeNews
export TASK_NAME=Fakeddit2way
export DATA_DIR=data/${TASK}/Fakeddit/2way
export OUTPUT_NAME=output
export MODEL=roberta
export MODEL_NAME=roberta-base
export MODEL_PATH=./models/${MODEL}/${MODEL_NAME}

export TRAIN_BATCH_SIZE=4
export EVAL_BATCH_SIZE=64
export DEFAULT_BATCH_SIZE=8
export DEFAULT_SAVE_STEPS=1000
export SAVE_STEPS=$((${DEFAULT_BATCH_SIZE}/${TRAIN_BATCH_SIZE}*${DEFAULT_SAVE_STEPS}/2))
export DEFAULT_MAX_SEQ_LENGTH=128
export MAX_SEQ_LENGTH=$((${DEFAULT_BATCH_SIZE}/${TRAIN_BATCH_SIZE}*${DEFAULT_MAX_SEQ_LENGTH}/2))


export STAGE_NUM=1
export NEXT_STAGE_NUM=9

python ./examples/run_classifier.py \
    --model_type ${MODEL} \
    --model_name_or_path ${MODEL_NAME} \
    --task_name ${TASK_NAME} \
    --do_lower_case \
    --data_dir ${DATA_DIR} \
    --max_seq_length ${DEFAULT_MAX_SEQ_LENGTH} \
    --per_gpu_train_batch_size ${TRAIN_BATCH_SIZE}   \
    --per_gpu_eval_batch_size ${EVAL_BATCH_SIZE}   \
    --per_gpu_test_batch_size ${EVAL_BATCH_SIZE}   \
    --per_gpu_pred_batch_size ${EVAL_BATCH_SIZE}   \
    --learning_rate 1e-5 \
    --weight_decay 0.0001 \
    --num_train_epochs 1.0 \
    --output_dir ${OUTPUT_NAME}/${TASK}/${TASK_NAME}-${MODEL_NAME}/stage_${NEXT_STAGE_NUM} \
    --save_steps ${DEFAULT_SAVE_STEPS} \
    --overwrite_cache \
    --eval_all_checkpoints \
    --do_test \
#    --do_pred \
#    --do_eval \
#    --do_train \
