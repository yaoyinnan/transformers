#!/usr/bin/env bash
python src/transformers/convert_roberta_original_pytorch_checkpoint_to_pytorch.py \
    --roberta_checkpoint_path models/roberta/roberta.large.mnli \
    --pytorch_dump_folder_path models/roberta/roberta.large.mnli-pytorch \
    --classification_head
