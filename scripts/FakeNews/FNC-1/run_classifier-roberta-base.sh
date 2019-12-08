#!/usr/bin/env bash
set -eux

export TASK=FakeNews
export TASK_NAME=FNC-1
export DATA_DIR=data/${TASK}/${TASK_NAME}
export OUTPUT_NAME=output
export PREDICT_NAME=predict
export MODEL=roberta
export MODEL_NAME=roberta-base

export DEFAULT_BATCH_SIZE=8
export BATCH_SIZE=4
export DEFAULT_SAVE_STEPS=1000
export SAVE_STEPS=$((${DEFAULT_BATCH_SIZE}/${BATCH_SIZE}*${DEFAULT_SAVE_STEPS}/2))
export DEFAULT_MAX_SEQ_LENGTH=128
export MAX_SEQ_LENGTH=$((${DEFAULT_BATCH_SIZE}/${BATCH_SIZE}*${DEFAULT_MAX_SEQ_LENGTH}/2))

export STAGE_NUM=2
export NEXT_STAGE_NUM=12

python ./examples/run_classifier.py \
    --model_type ${MODEL} \
    --model_name_or_path ${MODEL_NAME} \
    --task_name ${TASK_NAME} \
    --do_lower_case \
    --data_dir ${DATA_DIR} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --per_gpu_train_batch_size ${BATCH_SIZE}   \
    --per_gpu_eval_batch_size ${BATCH_SIZE}   \
    --per_gpu_predict_batch_size ${BATCH_SIZE}   \
    --learning_rate 1e-5 \
    --weight_decay 0.00001 \
    --num_train_epochs 10.0 \
    --output_dir ${OUTPUT_NAME}/${TASK}/${TASK_NAME}-${MODEL_NAME}/stage_${NEXT_STAGE_NUM} \
    --save_steps ${DEFAULT_SAVE_STEPS} \
    --predict_file ${PREDICT_NAME}/${TASK}/${TASK_NAME}-${MODEL_NAME}/stage_${NEXT_STAGE_NUM}/result.csv \
    --do_eval \
    --do_train \
#    --do_predict \
