# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for multimodal multiclass prediction on MM-IMDB dataset."""

import argparse
import glob
import json
import logging
import os
import random
import csv
import shutil

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import sys

o_path = os.getcwd()
sys.path.append(o_path)

from src.transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertModel,
    AlbertTokenizer,
    BertConfig,
    BertModel,
    BertTokenizer,
    DistilBertConfig,
    DistilBertModel,
    DistilBertTokenizer,
    MMBTConfig,
    MMBTForClassification,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
    XLMConfig,
    XLMModel,
    XLMTokenizer,
    XLNetConfig,
    XLNetModel,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)
from utils_multimodal import ImageEncoder, collate_fn, get_image_transforms, get_label_frequencies, \
    my_tasks_num_labels, my_processors, my_output_modes

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig, DistilBertConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetModel, XLNetTokenizer),
    "xlm": (XLMConfig, XLMModel, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertModel, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, criterion):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    # 计算最大steps
    import math
    args.all_steps = math.ceil((len(train_dataset) * args.num_train_epochs / args.per_gpu_train_batch_size))

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # # Check if continuing training from a checkpoint
    # if os.path.exists(args.model_name_or_path):
    #     # set global_step to gobal_step of last saved checkpoint from model path
    #     global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
    #     epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
    #     steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
    #
    #     logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #     logger.info("  Continuing training from epoch %d", epochs_trained)
    #     logger.info("  Continuing training from global step %d", global_step)
    #     logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    best_f1, n_no_improve = 0, 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            labels = batch[5]
            inputs = {
                "input_ids": batch[0],
                "input_modal": batch[2],
                "attention_mask": batch[1],
                "modal_start_tokens": batch[3],
                "modal_end_tokens": batch[4],
            }
            outputs = model(**inputs)
            logits = outputs[0]  # model outputs are always tuple in transformers (see doc)
            loss = criterion(logits, labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, criterion)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and (
                        global_step % args.save_steps == 0 or global_step == args.all_steps):
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training

                    torch.save(model_to_save.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))
                    # model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        # TODO 若训练超过规定的训练epochs的次数时，评估指标已不再提升，则停止训练。由于牵扯到evaluate，故先注释掉。
        # if args.local_rank == -1:
        #     result = evaluate(args, model, tokenizer, criterion)
        #     if results["micro_f1"] > best_f1:
        #         best_f1 = results["micro_f1"]
        #         n_no_improve = 0
        #     else:
        #         n_no_improve += 1
        #
        #     if n_no_improve > args.patience:
        #         train_iterator.close()
        #         break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, dataset, tokenizer, criterion, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = os.path.join(args.output_dir, prefix)
    eval_dataset = dataset

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn
    )

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            batch = tuple(t.to(args.device) for t in batch)
            labels = batch[5]
            inputs = {
                "input_ids": batch[0],
                "input_modal": batch[2],
                "attention_mask": batch[1],
                "modal_start_tokens": batch[3],
                "modal_end_tokens": batch[4],
            }
            outputs = model(**inputs)
            logits = outputs[0]  # model outputs are always tuple in transformers (see doc)
            tmp_eval_loss = criterion(logits, labels)
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = torch.sigmoid(logits).detach().cpu().numpy() > 0.5
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, torch.sigmoid(logits).detach().cpu().numpy() > 0.5, axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    result = {
        "loss": eval_loss,
        "macro_f1": f1_score(out_label_ids, preds, average="macro"),
        "micro_f1": f1_score(out_label_ids, preds, average="micro"),
        "score_name": "macro_f1",
    }

    output_eval_file = os.path.join(eval_output_dir, "eval_report.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        writer.write("%s = %s\n" % ("loss", str(eval_loss)))

    return result


def test(args, model, dataset, tokenizer, criterion, prefix=""):
    # Loop to handle MNLI double testing (matched, mis-matched)
    test_output_dir = os.path.join(args.output_dir, prefix)
    test_dataset = dataset

    if not os.path.exists(test_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(test_output_dir)

    args.test_batch_size = args.per_gpu_test_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset, sampler=test_sampler, batch_size=args.test_batch_size, collate_fn=collate_fn
    )

    # multi-gpu test
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Test!
    logger.info("***** Running testing {} *****".format(prefix))
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.test_batch_size)
    test_loss = 0.0
    nb_test_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            batch = tuple(t.to(args.device) for t in batch)
            labels = batch[5]
            inputs = {
                "input_ids": batch[0],
                "input_modal": batch[2],
                "attention_mask": batch[1],
                "modal_start_tokens": batch[3],
                "modal_end_tokens": batch[4],
            }
            outputs = model(**inputs)
            logits = outputs[0]  # model outputs are always tuple in transformers (see doc)
            tmp_test_loss = criterion(logits, labels)
            test_loss += tmp_test_loss.mean().item()
        nb_test_steps += 1
        if preds is None:
            preds = torch.sigmoid(logits).detach().cpu().numpy() > 0.5
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, torch.sigmoid(logits).detach().cpu().numpy() > 0.5, axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

    test_loss = test_loss / nb_test_steps
    result = {
        "loss": test_loss,
        "macro_f1": f1_score(out_label_ids, preds, average="macro"),
        "micro_f1": f1_score(out_label_ids, preds, average="micro"),
    }

    output_test_file = os.path.join(test_output_dir, "test_report_1.txt")
    with open(output_test_file, "w") as writer:
        logger.info("***** Test results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def load_examples(args, tokenizer, evaluate="train"):
    data_path = args.data_dir
    transforms = get_image_transforms()
    image_dir = args.image_dir
    data_processor = my_processors[args.task_name](image_dir, tokenizer, transforms,
                                                   args.max_seq_length - args.num_image_embeds - 2)

    dataset = []
    data_tensor = []
    if evaluate == "train":
        dataset, data_tensor = data_processor.get_train_examples(data_path)
    elif evaluate == "dev":
        dataset, data_tensor = data_processor.get_dev_examples(data_path)
    elif evaluate == "test":
        dataset, data_tensor = data_processor.get_test_examples(data_path)

    return dataset, data_tensor


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--num_image_embeds", default=1, type=int, help="Number of Image Embeddings from the Image Encoder"
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run test on the test set.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run predict on the predict set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--per_gpu_test_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        "--per_gpu_pred_batch_size", default=8, type=int, help="Batch size per GPU/CPU for predicting.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument("--patience", default=5, type=int, help="Patience for Early Stopping.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_workers", type=int, default=8, help="number of worker threads for dataloading")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--image_model",
        type=str,
        help="image model path",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        help="image dir",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1

    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Setup model
    labels = my_processors[args.task_name].labels
    num_labels = my_tasks_num_labels[args.task_name]
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    transformer_config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    transformer = model_class.from_pretrained(
        args.model_name_or_path, config=transformer_config, cache_dir=args.cache_dir if args.cache_dir else None
    )
    img_encoder = ImageEncoder(args)
    config = MMBTConfig(transformer_config, num_labels=num_labels)
    model = MMBTForClassification(config, transformer, img_encoder)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset, train_tensor = load_examples(args, tokenizer, evaluate="train")
        label_frequences = get_label_frequencies(train_dataset)
        label_frequences = [label_frequences[l] for l in labels]
        label_weights = (torch.tensor(label_frequences, device=args.device, dtype=torch.float) /
                         len(train_dataset)) ** -1
        criterion = nn.BCEWithLogitsLoss(pos_weight=label_weights)
        global_step, tr_loss = train(args, train_tensor, model, tokenizer, criterion)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    # if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    #     # Create output directory if needed
    #     if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
    #         os.makedirs(args.output_dir)
    #
    #     logger.info("Saving model checkpoint to %s", args.output_dir)
    #     # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    #     # They can then be reloaded using `from_pretrained()`
    #     model_to_save = (
    #         model.module if hasattr(model, "module") else model
    #     )  # Take care of distributed/parallel training
    #     torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, WEIGHTS_NAME))
    #     tokenizer.save_pretrained(args.output_dir)
    #
    #     # Good practice: save your training arguments together with the trained model
    #     torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    #
    #     # Load a trained model and vocabulary that you have fine-tuned
    #     model = MMBTForClassification(config, transformer, img_encoder)
    #     model.load_state_dict(torch.load(os.path.join(args.output_dir, WEIGHTS_NAME)))
    #     tokenizer = tokenizer_class.from_pretrained(args.output_dir)
    #     model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = []
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        eval_dataset, eval_tensor = load_examples(args, tokenizer, evaluate="dev")
        label_frequences = get_label_frequencies(eval_dataset)
        label_frequences = [label_frequences[l] for l in labels]
        label_weights = (torch.tensor(label_frequences, device=args.device, dtype=torch.float) / len(
            eval_dataset)) ** -1
        criterion = nn.BCEWithLogitsLoss(pos_weight=label_weights)

        checkpoint_score_max = 0
        checkpoint_score_max_checkpoint = ""
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = ("checkpoint-" + str(checkpoint.split('-')[-1])) if checkpoint.find('checkpoint') != -1 else ""
            model_path = os.path.join(args.output_dir, prefix)
            model = MMBTForClassification(config, transformer, img_encoder)
            model.load_state_dict(torch.load(os.path.join(checkpoint, WEIGHTS_NAME)))
            model.to(args.device)
            tokenizer = tokenizer_class.from_pretrained(model_path, do_lower_case=args.do_lower_case)
            result = evaluate(args=args, model=model, dataset=eval_tensor, tokenizer=tokenizer, criterion=criterion,
                              prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)
            checkpoint_score = result[result['score_name_' + global_step] + "_" + global_step]
            if checkpoint_score >= checkpoint_score_max:
                checkpoint_score_max = checkpoint_score
                checkpoint_score_max_checkpoint = prefix
        best_checkpoint_path = os.path.join(args.output_dir, "checkpoint-best")
        shutil.copytree(os.path.join(args.output_dir, checkpoint_score_max_checkpoint), best_checkpoint_path)

    # testing
    results = {}
    if args.do_test and args.local_rank in [-1, 0]:
        test_dataset, test_tensor = load_examples(args, tokenizer, evaluate="test")
        label_frequences = get_label_frequencies(test_dataset)
        label_frequences = [label_frequences[l] for l in labels]
        label_weights = (torch.tensor(label_frequences, device=args.device, dtype=torch.float) / len(
            test_dataset)) ** -1
        criterion = nn.BCEWithLogitsLoss(pos_weight=label_weights)
        checkpoint = os.path.join(args.output_dir, "checkpoint-best")
        logger.info("Predicting the following best checkpoint: %s", checkpoint)
        prefix = os.path.join("checkpoint-best", "test")
        model_path = os.path.join(args.output_dir, "checkpoint-best")
        model = MMBTForClassification(config, transformer, img_encoder)
        model.load_state_dict(torch.load(os.path.join(checkpoint, WEIGHTS_NAME)))
        model.to(args.device)
        tokenizer = tokenizer_class.from_pretrained(model_path, do_lower_case=args.do_lower_case)
        test(args=args, model=model, dataset=test_tensor, tokenizer=tokenizer, criterion=criterion, prefix=prefix)

    # # predicting
    # results = {}
    # if args.do_predict and args.local_rank in [-1, 0]:
    #     model_path = args.output_dir + "/checkpoint-best"
    #     tokenizer = tokenizer_class.from_pretrained(model_path, do_lower_case=args.do_lower_case)
    #     prefix = os.path.join("checkpoint-best", "test")
    #     checkpoint = args.output_dir + "/checkpoint-best"
    #     logger.info("Predicting the following best checkpoint: %s", checkpoint)
    #     model = model_class.from_pretrained(checkpoint)
    #     model.to(args.device)
    #     predict(args, model, tokenizer, criterion, prefix=prefix)

    return results


if __name__ == "__main__":
    main()
