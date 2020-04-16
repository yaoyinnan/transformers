python examples/multimodal/run_multimodal.py --model_type roberta --model_name_or_path /model/yaoyinnan/roberta-base --image_model /model/yaoyinnan/resnet/resnet152.pth --task_name Fakedditmm2way --data_dir /data/yaoyinnan/Fakeddit/2way --image_dir /data/yaoyinnan/Fakeddit/images --output_dir /output/Fakedditmm2way-bert-base-cased/stage_1 --max_seq_len 48 --per_gpu_train_batch_size 1 --per_gpu_eval_batch_size 2 --per_gpu_test_batch_size 2 --save_steps 1000 --num_image_embeds 3 --num_train_epochs 1 --eval_all_checkpoints --do_eval --do_train --do_test