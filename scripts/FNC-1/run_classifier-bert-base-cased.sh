#!/usr/bin/env bash
export DATA_DIR=data/FNC-1
export TASK_NAME=FNC-1
export OUTPUT_NAME=output
export PREDICT_NAME=predict
export MODEL=bert
export MODEL_NAME=bert-base-cased
export STAGE_NUM=1
export NEXT_STAGE_NUM=3-2

python ./examples/run_classifier.py \
    --model_type ${MODEL} \
    --model_name_or_path ${MODEL_NAME} \
    --task_name ${TASK_NAME} \
    --do_lower_case \
    --data_dir ${DATA_DIR} \
    --max_seq_length 512 \
    --per_gpu_train_batch_size 1   \
    --per_gpu_eval_batch_size 1   \
    --per_gpu_predict_batch_size 1   \
    --learning_rate 5e-5 \
    --weight_decay 0.001 \
    --num_train_epochs 10.0 \
    --output_dir ${OUTPUT_NAME}/${TASK_NAME}-${MODEL_NAME}/stage_${STAGE_NUM} \
    --save_steps 5000 \
    --predict_file ${PREDICT_NAME}/${TASK_NAME}-${MODEL_NAME}/stage_${STAGE_NUM}/result.csv \
    --do_predict \
#    --do_eval \
#    --do_train \
