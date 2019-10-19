#!/usr/bin/env bash
export DATA_DIR=data/OffensEval/Task2
export TASK_NAME=OffensEvalTask2
export OUTPUT_NAME=output
export PREDICT_NAME=predict
export MODEL=bert
export MODEL_NAME=bert-base-cased
export STAGE_NUM=2-2
export NEXT_STAGE_NUM=3-2

python ./examples/run_classifier.py \
    --model_type ${MODEL} \
    --model_name_or_path ${MODEL_NAME} \
    --task_name ${TASK_NAME} \
    --do_lower_case \
    --data_dir ${DATA_DIR} \
    --max_seq_length 128 \
    --per_gpu_train_batch_size 8   \
    --per_gpu_eval_batch_size 8   \
    --per_gpu_predict_batch_size 8   \
    --learning_rate 5e-5 \
    --weight_decay 0.0001 \
    --num_train_epochs 8.0 \
    --output_dir ${OUTPUT_NAME}/${TASK_NAME}-${MODEL_NAME} \
    --save_steps 1000 \
    --do_eval \
    --do_train \
#    --predict_file ${PREDICT_NAME}/${TASK_NAME}-${MODEL_NAME}/result.csv