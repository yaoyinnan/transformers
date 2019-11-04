#!/usr/bin/env bash
export DATA_DIR=data/FNC-1
export TASK_NAME=FNC-1
export OUTPUT_NAME=output
export PREDICT_NAME=predict
export MODEL=bert
export MODEL_NAME=bert-base-cased

export DEFAULT_BATCH_SIZE=8
export BATCH_SIZE=8
export DEFAULT_SAVE_STEPS=1000
export SAVE_STEPS=$((${DEFAULT_BATCH_SIZE}/${BATCH_SIZE}*${DEFAULT_SAVE_STEPS}))
export DEFAULT_MAX_SEQ_LENGTH=128
export MAX_SEQ_LENGTH=$((${DEFAULT_BATCH_SIZE}/${BATCH_SIZE}*${DEFAULT_MAX_SEQ_LENGTH}))

export STAGE_NUM=1
export NEXT_STAGE_NUM=2

python ./examples/run_classifier.py \
    --model_type ${MODEL} \
    --model_name_or_path ${OUTPUT_NAME}/${TASK_NAME}-${MODEL_NAME}/stage_${STAGE_NUM} \
    --task_name ${TASK_NAME} \
    --do_lower_case \
    --data_dir ${DATA_DIR} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --per_gpu_train_batch_size ${BATCH_SIZE}   \
    --per_gpu_eval_batch_size ${BATCH_SIZE}   \
    --per_gpu_predict_batch_size ${BATCH_SIZE}   \
    --learning_rate 2e-5 \
    --weight_decay 0.00005 \
    --num_train_epochs 10.0 \
    --output_dir ${OUTPUT_NAME}/${TASK_NAME}-${MODEL_NAME}/stage_${NEXT_STAGE_NUM} \
    --save_steps ${SAVE_STEPS} \
    --predict_file ${PREDICT_NAME}/${TASK_NAME}-${MODEL_NAME}/stage_${NEXT_STAGE_NUM}/result.csv \
    --do_train \
    --do_eval \
#    --do_predict \
