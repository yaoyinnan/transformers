#!/usr/bin/env bash
python transformers/convert_albert_tf_checkpoint_to_pytorch.py \
    --tf_checkpoint_path=models/albert/albert-large-zh \
    --bert_config_file=models/albert/albert-large-zh/albert_config_large.json \
    --pytorch_dump_path=models/albert/albert-large-zh/pytorch_model.bin \
    --share_type=all