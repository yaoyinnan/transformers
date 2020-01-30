# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" fnews processors and helpers """

import logging
import os

from .utils import DataProcessor, InputExample, InputFeatures
from ...file_utils import is_tf_available

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def my_convert_examples_to_features(examples,
                                    evaluate,
                                    tokenizer,
                                    max_length=512,
                                    task=None,
                                    label_list=None,
                                    output_mode=None,
                                    pad_on_left=False,
                                    pad_token=0,
                                    pad_token_segment_id=0,
                                    mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: fnews task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.
        :param mask_padding_with_zero:
        :param pad_token_segment_id:
        :param pad_token:
        :param pad_on_left:
        :param output_mode:
        :param label_list:
        :param task:
        :param max_length:
        :param examples:
        :param tokenizer:
        :param evaluate:

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = my_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = my_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    guids = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)
        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                            max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                            max_length)

        label = None
        if evaluate == "train" or evaluate == "dev":
            if output_mode == "classification":
                label = label_map[example.label]
            elif output_mode == "regression":
                label = float(example.label)
            else:
                raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            if evaluate == "train" or evaluate == "dev":
                logger.info("label: %s (id = %d)" % (example.label, label))
        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label=label))

        guids.append(example.guid)

    if is_tf_available() and is_tf_dataset:
        def gen():
            for ex in features:
                yield ({'input_ids': ex.input_ids,
                        'attention_mask': ex.attention_mask,
                        'token_type_ids': ex.token_type_ids},
                       ex.label)

        return tf.data.Dataset.from_generator(gen,
                                              ({'input_ids': tf.int32,
                                                'attention_mask': tf.int32,
                                                'token_type_ids': tf.int32},
                                               tf.int64),
                                              ({'input_ids': tf.TensorShape([None]),
                                                'attention_mask': tf.TensorShape([None]),
                                                'token_type_ids': tf.TensorShape([None])},
                                               tf.TensorShape([])))

    if evaluate == "train":
        return features
    elif evaluate == "dev" or evaluate == "predict":
        return features, guids


class FnewsProcessor(DataProcessor):
    """Processor for the Fnews data set (My version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_predict_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "predict.tsv")), "predict")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if set_type == "train" or set_type == "dev":
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                label = line[1]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            elif set_type == "predict":
                guid = line[0]
                text_a = line[1]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=None))

        return examples


class OffensEval2019Task1Processor(DataProcessor):
    """Processor for the OffensEval2019Task1 data set (My version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_predict_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "predict")

    def get_labels(self):
        """See base class."""
        return ["OFF", "NOT"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "train" or set_type == "dev":
                text_a = line[1]
                label = line[0]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            elif set_type == "predict":
                text_a = line[1]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=None))

        return examples


class OffensEval2019Task2Processor(DataProcessor):
    """Processor for the OffensEval2019Task2 data set (My version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_predict_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "predict")

    def get_labels(self):
        """See base class."""
        return ["TIN", "UNT"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "train" or set_type == "dev":
                text_a = line[1]
                label = line[0]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            elif set_type == "predict":
                text_a = line[1]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=None))

        return examples


class OffensEval2019Task3Processor(DataProcessor):
    """Processor for the OffensEval2019Task3 data set (My version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_predict_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "predict")

    def get_labels(self):
        """See base class."""
        return ["IND", "OTH", "GRP"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "train" or set_type == "dev":
                text_a = line[1]
                label = line[0]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            elif set_type == "predict":
                text_a = line[1]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=None))

        return examples


class OffensEval2020Task1EnglishProcessor(DataProcessor):
    """Processor for the OffensEval2020Task1English data set (My version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "88train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_predict_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "predict")

    def get_labels(self):
        """See base class."""
        return ["OFF", "NOT"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "train" or set_type == "dev":
                text_a = line[1]
                label = line[0]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            elif set_type == "predict":
                text_a = line[1]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=None))

        return examples


class FNC1Processor(DataProcessor):
    """Processor for the FNC1 data set (My version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "eda_train_stances_stage2_RD.csv"), '"'),
            self._read_csv(os.path.join(data_dir, "train_bodies.csv"), '"'),
            "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "test_stances_stage2.csv"), '"'),
            self._read_csv(os.path.join(data_dir, "test_bodies.csv"), '"'),
            "dev")

    def get_predict_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "test_stances_stage2.csv"), '"'),
            self._read_csv(os.path.join(data_dir, "test_bodies.csv"), '"'),
            "predict")

    def get_labels(self):
        """See base class."""
        return ["discuss", "agree", "disagree"]

    def _create_examples(self, stance_lines, body_lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        # label_list = [860, 2630, 2965, 3145]  # no-balance
        # label_list = [2400, 2400, 2400, 2400] # balance 800
        # label_list = [0, 0, 0, 1600]  # balance 1600(disagree = double)
        # label_list = [0, 0, 0, 0]  # balance 800(disagree = one)
        # label_list = [0, 0, 6000, 7200]  # balance 800(disagree = one)

        # label_list = [0, 0, 1600]  # 三份disagree
        # max_num = 2400  # 三份disagree

        # label_list = [0, 3200, 5600]  # 二份agree + 八份disagree
        # max_num = 6400  # 二份agree + 八份disagree

        # label_list = [0, 0, 0]  # eda 16800 * 3
        # max_num = 16800  # eda 16800 * 3

        for (i, line) in enumerate(stance_lines):
            # if i >= len(stance_lines):
            #     break
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            headline = line[0]
            body_id = int(line[1])
            label = line[2]

            # if set_type == "train":
            #     labels = self.get_labels()
            #     flag = False
            #     for label_index in range(len(label_list)):
            #         if label == labels[label_index]:
            #             if label_list[label_index] >= max_num:
            #                 flag = True
            #                 break
            #             label_list[label_index] += 1
            #     if flag:
            #         continue
            #
            #     if label_list == [max_num for n in range(len(self.get_labels()))]:
            #         break
            #
            # elif set_type == "predict" or set_type == "dev":
            #     pass

            def search(body_lines, body_id):
                left_i = 0
                right_i = len(body_lines)
                tar_i = int((right_i + left_i) / 2)
                while left_i < right_i:
                    tar_body_id = int(body_lines[tar_i][0])
                    if body_id < tar_body_id:
                        right_i = tar_i
                        tar_i = int((right_i + left_i) / 2)
                    elif body_id > tar_body_id:
                        left_i = tar_i
                        tar_i = int((right_i + left_i) / 2)
                    elif body_id == tar_body_id:
                        return body_lines[tar_i][1]

            article_body = search(body_lines=body_lines, body_id=body_id)
            text_a = headline + article_body
            # text_a = self.preprocess(text_a)
            # 去停用词（基本都是负作用）
            # text_a = self.stop_word(text_a, "english")

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

            # # 三份disagree
            # if label == self.get_labels()[2]:
            #     for i in range(0, 2):
            #         examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

            # # 二份agree
            # if label == self.get_labels()[1]:
            #     for i in range(0, 1):
            #         examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            # # 八份disagree
            # if label == self.get_labels()[2]:
            #     for i in range(0, 7):
            #         examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        # print(examples)
        return examples


class WSDMFakeNewsProcessor(DataProcessor):
    """Processor for the WSDMFakeNews data set (My version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train.csv"), '"'), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "dev.csv"), '"'), "dev")

    def get_predict_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "test.csv"), '"'), "predict")

    def get_labels(self):
        """See base class."""
        return ["agreed", "disagreed", "unrelated"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = line[0]
            if set_type == "train" or set_type == "dev":
                if i >= 3000:
                    break
                text_a = line[3] + line[4]
                text_a = self.preprocess(text_a)
                label = line[7]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            elif set_type == "predict":
                text_a = line[3] + line[4]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=None))
        return examples


class LIARProcessor(DataProcessor):
    """Processor for the LIAR data set (My version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_predict_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "predict")

    def get_labels(self):
        """See base class."""
        return ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = line[0]
            if set_type == "train" or set_type == "dev":
                # if i >= 3000:
                #     break
                label = line[1]
                text_a = line[2]
                text_a = self.preprocess(text_a)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            elif set_type == "predict":
                text_a = line[2]
                text_a = self.preprocess(text_a)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=None))
        return examples


class FEVERProcessor(DataProcessor):
    """Processor for the FEVER data set (My version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "shared_task_dev.json")), "dev")

    def get_predict_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "shared_task_test.json")), "predict")

    def get_labels(self):
        """See base class."""
        return ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = line['id']
            if set_type == "train" or set_type == "dev":
                # if i >= 3000:
                #     break
                label = line['label']
                text_a = line['claim']
                text_a = self.preprocess(text_a)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            elif set_type == "predict":
                text_a = line['claim']
                text_a = self.preprocess(text_a)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=None))
        return examples


class FakedditProcessor(DataProcessor):
    """Processor for the Fakeddit data set (My version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv"), '"'), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "validate.tsv"), '"'), "dev")

    def get_predict_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv"), '"'), "predict")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = line[0]
            if set_type == "train":
                if i >= 10000:
                    break
                label = line[13]
                text_a = line[12]
                text_a = self.preprocess(text_a)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            elif set_type == "dev":
                label = line[13]
                text_a = line[12]
                text_a = line[12]
                text_a = self.preprocess(text_a)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            elif set_type == "predict":
                text_a = line[12]
                text_a = self.preprocess(text_a)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=None))
        return examples


my_tasks_num_labels = {
    "fnews": 2,
    "fnc-1": 3,
    "wsdm-fakenews": 3,
    "liar": 6,
    "fever": 3,
    "fakeddit": 2,
    "offenseval2019task1": 2,
    "offenseval2019task2": 2,
    "offenseval2019task3": 3,
    "offenseval2020task1english": 2,
}

my_processors = {
    "fnews": FnewsProcessor,
    "fnc-1": FNC1Processor,
    "wsdm-fakenews": WSDMFakeNewsProcessor,
    "liar": LIARProcessor,
    "fever": FEVERProcessor,
    "fakeddit": FakedditProcessor,
    "offenseval2019task1": OffensEval2019Task1Processor,
    "offenseval2019task2": OffensEval2019Task2Processor,
    "offenseval2019task3": OffensEval2019Task3Processor,
    "offenseval2020task1english": OffensEval2020Task1EnglishProcessor,
}

my_output_modes = {
    "fnews": "classification",
    "fnc-1": "classification",
    "wsdm-fakenews": "classification",
    "liar": "classification",
    "fever": "classification",
    "fakeddit": "classification",
    "offenseval2019task1": "classification",
    "offenseval2019task2": "classification",
    "offenseval2019task3": "classification",
    "offenseval2020task1english": "classification",
}
