#!/usr/bin/env bash

export TASK=FakeNews
export TASK_NAME=Fakedditmm2way
export DATA_DIR=data/${TASK}/Fakeddit/2way
export IMAGE_DIR=data/${TASK}/Fakeddit/images
export OUTPUT_NAME=output
export MODEL=bert
export MODEL_NAME=bert-base-cased

export TRAIN_BATCH_SIZE=144
export EVAL_BATCH_SIZE=1024
export DEFAULT_BATCH_SIZE=8
export DEFAULT_SAVE_STEPS=1000
export SAVE_STEPS=$((${DEFAULT_BATCH_SIZE}/${TRAIN_BATCH_SIZE}*${DEFAULT_SAVE_STEPS}/2))
export DEFAULT_MAX_SEQ_LENGTH=48
export MAX_SEQ_LENGTH=$((${DEFAULT_BATCH_SIZE}/${TRAIN_BATCH_SIZE}*${DEFAULT_MAX_SEQ_LENGTH}/2))

export STAGE_NUM=1
export NEXT_STAGE_NUM=3

python ./examples/multimodal-batch/run_multimodal.py \
    --model_type ${MODEL} \
    --model_name_or_path ${MODEL_NAME} \
    --image_model ./models/resnet/resnet152.pth \
    --task_name ${TASK_NAME} \
    --data_dir ${DATA_DIR} \
    --image_dir ${IMAGE_DIR} \
    --output_dir ${OUTPUT_NAME}/${TASK}/${TASK_NAME}-${MODEL_NAME}/stage_${NEXT_STAGE_NUM} \
    --max_seq_len ${DEFAULT_MAX_SEQ_LENGTH} \
    --per_gpu_train_batch_size ${TRAIN_BATCH_SIZE}   \
    --per_gpu_eval_batch_size ${EVAL_BATCH_SIZE}   \
    --per_gpu_test_batch_size ${EVAL_BATCH_SIZE}   \
    --save_steps ${DEFAULT_SAVE_STEPS} \
    --num_image_embeds 3 \
    --num_train_epochs 1 \
    --eval_all_checkpoints  \
    --no_cuda \
    --do_test  \
    --do_eval  \
    --do_train \
    --do_bitahub  \
#    --model_name_or_path ${OUTPUT_NAME}/${TASK}/${TASK_NAME}-${MODEL_NAME}/stage_${NEXT_STAGE_NUM}/checkpoint-20924 \
#    --gradient_accumulation_steps 20 \

