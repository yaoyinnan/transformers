#!/usr/bin/env bash
export DATA_DIR=data/fnews
export TASK_NAME=FNews
export OUTPUT_NAME=output
export PREDICT_NAME=predict
export MODEL=bert
export MODEL_NAME=bert-base-chinese
export STAGE_NUM=2-2
export NEXT_STAGE_NUM=3-4

python ./examples/run_classifier.py \
    --model_type ${MODEL} \
    --model_name_or_path ${OUTPUT_NAME}/${TASK_NAME}-${MODEL_NAME}/stage_${STAGE_NUM} \
    --task_name ${TASK_NAME} \
    --do_train \
    --do_eval \
    --do_predict \
    --do_lower_case \
    --data_dir ${DATA_DIR} \
    --max_seq_length 128 \
    --per_gpu_train_batch_size 8   \
    --per_gpu_eval_batch_size 8   \
    --per_gpu_predict_batch_size 8   \
    --learning_rate 1e-5 \
    --weight_decay 0.001 \
    --num_train_epochs 10.0 \
    --output_dir ${OUTPUT_NAME}/${TASK_NAME}-${MODEL_NAME}/stage_${NEXT_STAGE_NUM} \
    --save_steps 1000 \
    --predict_file ${PREDICT_NAME}/${TASK_NAME}-${MODEL_NAME}/stage_${NEXT_STAGE_NUM}/result.csv