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

import json
import os
import random
import csv
import json
import logging
import emoji
import re
import nltk
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

POOLING_BREAKDOWN = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2), 5: (5, 1), 6: (3, 2), 7: (7, 1), 8: (4, 2), 9: (3, 3)}

logger = logging.getLogger(__name__)


class ImageEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        # model = torchvision.models.resnet152(pretrained=True)
        model = torchvision.models.resnet152(pretrained=False)
        model.load_state_dict(torch.load(args.image_model))
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d(POOLING_BREAKDOWN[args.num_image_embeds])

    def forward(self, x):
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
        out = self.pool(self.model(x))
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out  # BxNx2048


def collate_fn(batch):
    lens = [len(row["sentence"]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)
    text_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        text_tensor[i_batch, :length] = input_row["sentence"]
        mask_tensor[i_batch, :length] = 1

    img_tensor = torch.stack([row["image"] for row in batch])
    tgt_tensor = torch.stack([row["label"] for row in batch])
    img_start_token = torch.stack([row["image_start_token"] for row in batch])
    img_end_token = torch.stack([row["image_end_token"] for row in batch])

    return text_tensor, mask_tensor, img_tensor, img_start_token, img_end_token, tgt_tensor


def get_image_transforms():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.46777044, 0.44531429, 0.40661017], std=[0.12221994, 0.12145835, 0.14380469], ),
        ]
    )


def get_label_frequencies(data):
    label_freqs = Counter()
    for row in data:
        label_freqs.update(row["label"])
    return label_freqs


class DataProcessor(Dataset):
    def __init__(self, images_dir, tokenizer, transforms, max_seq_length):
        self.data = []
        self.images_dir = images_dir
        self.tokenizer = tokenizer
        self.n_classes = len(self.labels)
        self.max_seq_length = max_seq_length

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    labels = []

    def to_tensor(self, item):
        sentence = torch.LongTensor(self.tokenizer.encode(item["text"], add_special_tokens=True))
        start_token, sentence, end_token = sentence[0], sentence[1:-1], sentence[-1]
        sentence = sentence[: self.max_seq_length]

        label = torch.zeros(self.n_classes)
        label[[self.labels.index(tgt) for tgt in item["label"]]] = 1

        image = Image.open(os.path.join(self.images_dir, item["img"])).convert("RGB")
        image = self.transforms(image)

        return {
            "image_start_token": start_token,
            "image_end_token": end_token,
            "sentence": sentence,
            "image": image,
            "label": label,
        }

    def get_tensor(self, data):
        tensor = []
        for item in data:
            tensor.append(self.to_tensor(item))
        return tensor

    def str_to_bool(self, str):
        return True if str.lower() == 'True' else False

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter=",", quotechar=quotechar))

    @classmethod
    def _read_json_normal(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = json.load(f)
            lines = []
            for index, item in enumerate(reader):
                lines.append(item)
            return lines

    @classmethod
    def _read_json_abnormal(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = f.read().split("\n")
            lines = []
            for line in reader:
                load_dict = json.loads(line)
                lines.append(load_dict)
            return lines

    def label_index_to_label(self, label_index_list):
        """将label_index列表转化为label列表"""
        label = self.labels
        label_list = []
        for i in label_index_list:
            label_list.append(label[i])

        return label_list

    def preprocess(self, text):
        """文本预处理"""
        # 替换表情
        text = emoji.demojize(text)

        # 转化为小写
        text = text.lower()

        # 去除数据中的非文本部分
        def scrub_words(text):
            """Basic cleaning of texts."""

            # 过滤不了\\ \ 中文（）还有————
            r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'  # 用户也可以在此进行自定义过滤字符
            # 者中规则也过滤不完全
            # r2 = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
            r2 = "[\s+\.\!\/_,^*(+\"\']+|[+——！，。？、~%……&*（）]+"
            # \\\可以过滤掉反向单杠和双杠，/可以过滤掉正向单杠和双杠，第一个中括号里放的是英文符号，第二个中括号里放的是中文符号，第二个中括号前不能少|，否则过滤不完全
            r3 = "[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[——！\\\\，。=？、：“”‘’《》【】￥……（）]+"
            # 去掉括号和括号内的所有内容
            r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"

            text = re.sub(r2, ' ', text)

            text = text.strip()
            return text

        text = scrub_words(text)

        # 词干提取(stemming)和词型还原(lemmatization)
        wnl = WordNetLemmatizer()
        text = wnl.lemmatize(text)

        return text

    def stop_word(self, text, lan="english"):
        """去停用词（基本都是负作用）"""
        # 分词
        pattern = r"""(?x)                   # set flag to allow verbose regexps 
                    	              (?:[A-Z]\.)+           # abbreviations, e.g. U.S.A. 
                    	              |\d+(?:\.\d+)?%?       # numbers, incl. currency and percentages 
                    	              |\w+(?:[-']\w+)*       # words w/ optional internal hyphens/apostrophe 
                    	              |\.\.\.                # ellipsis 
                    	              |(?:[.,;"'?():-_`])    # special characters with meanings 
                    	            """
        text = nltk.regexp_tokenize(text, pattern)

        # 去停止词
        text = [word for word in text if word not in stopwords.words(lan)]

        # 拼接
        text = ' '.join(text)

        return text

    def html(self, text):
        import re
        dr = re.compile(r'<[^>]+>', re.S)
        dd = dr.sub('', text)

        return dd


class FakedditMM2wayProcessor(DataProcessor):
    """Processor for the Fakeddit2way data set (My version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train_images_10000.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_images_10000.tsv"), '"'), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "validate_images_10000.tsv"), '"'), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_images_10000.tsv"), '"'), "test")

    def get_predict_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "predict.tsv"), '"'), "predict")

    labels = ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = line[0]
            if set_type == "train":
                label = line[13]
                text = line[12]
                text = self.preprocess(text)
                img = (line[2] if self.str_to_bool(line[6]) else "default") + ".jpg"
                examples.append({"guid": guid, "text": text, "label": label, "img": img})
            elif set_type == "dev":
                label = line[13]
                text = line[12]
                text = self.preprocess(text)
                img = (line[2] if self.str_to_bool(line[6]) else "default") + ".jpg"
                examples.append({"guid": guid, "text": text, "label": label, "img": img})
            elif set_type == "test":
                label = line[14]
                text = line[13]
                text = self.preprocess(text)
                img = (line[3] if self.str_to_bool(line[6]) else "default") + ".jpg"
                examples.append({"guid": guid, "text": text, "label": label, "img": img})
        return examples


class FakedditMM3wayProcessor(DataProcessor):
    """Processor for the Fakeddit3way data set (My version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train_100000.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_100000.tsv"), '"'), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "validate.tsv"), '"'), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv"), '"'), "test")

    def get_predict_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "predict.tsv"), '"'), "predict")

    labels = ["0", "1", "2"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = line[0]
            if set_type == "train":
                label = line[14]
                text = line[12]
                text = self.preprocess(text)
                img = (line[2] if self.str_to_bool(line[6]) else "default") + ".jpg"
                examples.append({"guid": guid, "text": text, "label": label, "img": img})
            elif set_type == "dev":
                label = line[14]
                text = line[12]
                text = self.preprocess(text)
                img = (line[2] if self.str_to_bool(line[6]) else "default") + ".jpg"
                examples.append({"guid": guid, "text": text, "label": label, "img": img})
            elif set_type == "test":
                label = line[15]
                text = line[13]
                text = self.preprocess(text)
                img = (line[3] if self.str_to_bool(line[6]) else "default") + ".jpg"
                examples.append({"guid": guid, "text": text, "label": label, "img": img})
        return examples, self.get_tensor(examples)


class FakedditMM5wayProcessor(DataProcessor):
    """Processor for the Fakeddit5way data set (My version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train_125290.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_125290.tsv"), '"'), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "validate.tsv"), '"'), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv"), '"'), "test")

    def get_predict_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "predict.tsv"), '"'), "predict")

    labels = ["0", "1", "2", "3", "4"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = line[0]
            if set_type == "train":
                label = line[15]
                text = line[12]
                text = self.preprocess(text)
                img = (line[2] if self.str_to_bool(line[6]) else "default") + ".jpg"
                examples.append({"guid": guid, "text": text, "label": label, "img": img})
            elif set_type == "dev":
                label = line[15]
                text = line[12]
                text = self.preprocess(text)
                img = (line[2] if self.str_to_bool(line[6]) else "default") + ".jpg"
                examples.append({"guid": guid, "text": text, "label": label, "img": img})
            elif set_type == "test":
                label = line[16]
                text = line[13]
                text = self.preprocess(text)
                img = (line[3] if self.str_to_bool(line[6]) else "default") + ".jpg"
                examples.append({"guid": guid, "text": text, "label": label, "img": img})
        return examples, self.get_tensor(examples)


class FakedditMMFineGrainedProcessor(DataProcessor):
    """Processor for the Fakeddit5way data set (My version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv"), '"'), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "validate.tsv"), '"'), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv"), '"'), "test")

    def get_predict_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "predict.tsv"), '"'), "predict")

    labels = ['nottheonion',
                'fakealbumcovers',
                'confusing_perspective',
                'pareidolia',
                'upliftingnews',
                'pic',
                'mildlyinteresting',
                'fakehistoryporn',
                'theonion',
                'photoshopbattles',
                'misleadingthumbnails',
                'usnews',
                'propagandaposters',
                'subredditsimulator',
                'usanews',
                'neutralnews',
                'satire',
                'savedyouaclick',
                'subsimulatorgpt2',
                'fakefacts',
                'waterfordwhispersnews']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = line[0]
            if set_type == "train":
                label = line[11]
                text = line[12]
                text = self.preprocess(text)
                img = (line[2] if self.str_to_bool(line[6]) else "default") + ".jpg"
                examples.append({"guid": guid, "text": text, "label": label, "img": img})
            elif set_type == "dev":
                label = line[11]
                text = line[12]
                text = self.preprocess(text)
                img = (line[2] if self.str_to_bool(line[6]) else "default") + ".jpg"
                examples.append({"guid": guid, "text": text, "label": label, "img": img})
            elif set_type == "test":
                label = line[12]
                text = line[13]
                text = self.preprocess(text)
                img = (line[3] if self.str_to_bool(line[6]) else "default") + ".jpg"
                examples.append({"guid": guid, "text": text, "label": label, "img": img})
        return examples, self.get_tensor(examples)


my_tasks_num_labels = {
    "fnews": 2,
    "fnc-1": 3,
    "wsdm-fakenews": 3,
    "liar": 6,
    "fever": 3,
    "Fakedditmm2way": 2,
    "Fakedditmm3way": 3,
    "Fakedditmm5way": 5,
    "Fakedditmmfinegrained": 21,
    "wuhan2019ncov": 2,
    "offenseval2019task1": 2,
    "offenseval2019task2": 2,
    "offenseval2019task3": 3,
    "offenseval2020task1english": 2,
}

my_processors = {
    # "fnews": FnewsProcessor,
    # "fnc-1": FNC1Processor,
    # "wsdm-fakenews": WSDMFakeNewsProcessor,
    # "liar": LIARProcessor,
    # "fever": FEVERProcessor,
    "Fakedditmm2way": FakedditMM2wayProcessor,
    "Fakedditmm3way": FakedditMM3wayProcessor,
    "Fakedditmm5way": FakedditMM5wayProcessor,
    "Fakedditmmfinegrained": FakedditMMFineGrainedProcessor,
    # "wuhan2019ncov": wuhan2019ncovProcessor,
    # "offenseval2019task1": OffensEval2019Task1Processor,
    # "offenseval2019task2": OffensEval2019Task2Processor,
    # "offenseval2019task3": OffensEval2019Task3Processor,
    # "offenseval2020task1english": OffensEval2020Task1EnglishProcessor,
}

my_output_modes = {
    "fnews": "classification",
    "fnc-1": "classification",
    "wsdm-fakenews": "classification",
    "liar": "classification",
    "fever": "classification",
    "Fakedditmm2way": "classification",
    "Fakedditmm3way": "classification",
    "Fakedditmm5way": "classification",
    "Fakedditmmfinegrained": "classification",
    "wuhan2019ncov": "classification",
    "offenseval2019task1": "classification",
    "offenseval2019task2": "classification",
    "offenseval2019task3": "classification",
    "offenseval2020task1english": "classification",
}
