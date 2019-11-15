#!/usr/bin/env bash
export DATA_DIR=data/OffensEval/Task2
export TASK_NAME=OffensEvalTask2
export OUTPUT_NAME=output
export PREDICT_NAME=predict
export MODEL=bert
export MODEL_NAME=bert-base-uncased
export STAGE_NUM=2-2
export NEXT_STAGE_NUM=1

python ./examples/run_classifier.py \
    --model_type ${MODEL} \
    --model_name_or_path ${MODEL_NAME} \
    --task_name ${TASK_NAME} \
    --do_lower_case \
    --data_dir ${DATA_DIR} \
    --max_seq_length 64 \
    --per_gpu_train_batch_size 16   \
    --per_gpu_eval_batch_size 16   \
    --per_gpu_predict_batch_size 16   \
    --learning_rate 2e-5 \
    --num_train_epochs 2.0 \
    --output_dir ${OUTPUT_NAME}/${TASK_NAME}-${MODEL_NAME}/stage_${NEXT_STAGE_NUM} \
    --predict_file ${PREDICT_NAME}/${TASK_NAME}-${MODEL_NAME}/result.csv  \
    --save_steps 1000 \
    --do_eval \
#    --do_train \
#    --weight_decay 0.001 \
