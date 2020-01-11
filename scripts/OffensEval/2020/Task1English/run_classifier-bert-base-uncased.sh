#!/usr/bin/env bash
export TASK=OffensEval
export TASK_NAME=OffensEval2020Task1English
export DATA_DIR=data/OffensEval/2020/Task1English
export OUTPUT_NAME=output
export PREDICT_NAME=predict
export MODEL=bert
export MODEL_NAME=bert-base-uncased
export STAGE_NUM=1-2
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
    --learning_rate 6e-5 \
    --weight_decay 0.001 \
    --num_train_epochs 10.0 \
    --output_dir ${OUTPUT_NAME}/${TASK}/${TASK_NAME}-${MODEL_NAME}/stage_${STAGE_NUM} \
    --save_steps 1000 \
    --do_predict \
    --predict_file ${PREDICT_NAME}/${TASK}/${TASK_NAME}-${MODEL_NAME}/stage_${STAGE_NUM}/result.csv
#    --do_eval \
#    --do_train \
