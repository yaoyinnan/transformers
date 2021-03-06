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

import csv
import sys
import logging

logger = logging.getLogger(__name__)

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn import metrics

    _has_sklearn = True
except (AttributeError, ImportError) as e:
    logger.warning("To use data.metrics please install scikit-learn. See https://scikit-learn.org/stable/index.html")
    _has_sklearn = False


def is_sklearn_available():
    return _has_sklearn


if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()


    def acc_and_f1_micro(preds, labels):
        acc = simple_accuracy(preds, labels)
        precision = metrics.precision_score(y_true=labels, y_pred=preds, average='micro')
        recall = metrics.recall_score(y_true=labels, y_pred=preds, average='micro')
        f1 = metrics.f1_score(y_true=labels, y_pred=preds, average='micro')
        return {
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "micro-f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }


    def acc_and_f1_macro(preds, labels):
        acc = simple_accuracy(preds, labels)
        precision = metrics.precision_score(y_true=labels, y_pred=preds, average='macro')
        recall = metrics.recall_score(y_true=labels, y_pred=preds, average='macro')
        f1 = metrics.f1_score(y_true=labels, y_pred=preds, average='macro')
        return {
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "macro-f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }


    def classification_report(preds, labels, target_names=None):
        f1 = metrics.f1_score(y_true=labels, y_pred=preds, average='macro')
        report = metrics.classification_report(y_true=labels, y_pred=preds, target_names=target_names, digits=4)
        return {
            "report": report,
            "score_name": "macro-f1",
            "macro-f1": f1,
        }


    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
            "score_name": "spearmanr",
        }


    def glue_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(labels)
        if task_name == "cola":
            return {"mcc": metrics.matthews_corrcoef(labels, preds)}
        elif task_name == "sst-2":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mrpc":
            return acc_and_f1_micro(preds, labels)
        elif task_name == "sts-b":
            return pearson_and_spearman(preds, labels)
        elif task_name == "qqp":
            return acc_and_f1_micro(preds, labels)
        elif task_name == "mnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mnli-mm":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "qnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "rte":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "wnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "hans":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)


    def xnli_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(labels)
        if task_name == "xnli":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)


    def my_compute_metrics(task_name, preds, labels, target_names=None):
        assert len(preds) == len(labels)
        if task_name == "fnews":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "fnc-1":
            return classification_report(preds, labels, target_names)
        elif task_name == "wsdm-fakenews":
            return classification_report(preds, labels, target_names)
        elif task_name == "liar":
            return classification_report(preds, labels, target_names)
        elif task_name == "fever":
            return classification_report(preds, labels, target_names)
        elif task_name in ["fakeddit2way", "fakeddit3way", "fakeddit5way", "fakedditfinegrained"]:
            return classification_report(preds, labels, target_names)
        elif task_name in ["fakeddit2waytoliar", "fakeddit3waytoliar",
                           "fakeddit2waytofakenewsnetgossipcop", "fakeddit2waytofakenewsnetpolitifact",
                           "fakeddit3waytofever"]:
            return classification_report(preds, labels, target_names)
        elif task_name in ["fakedditstance"]:
            return pearson_and_spearman(preds, labels)
        elif task_name in ["fakenewsnet", "fakenewsnetgossipcop", "fakenewsnetpolitifact"]:
            return classification_report(preds, labels, target_names)
        elif task_name == "wuhan2019ncov":
            return classification_report(preds, labels, target_names)
        elif task_name == "covid19":
            return classification_report(preds, labels, target_names)
        elif task_name in ["offenseval2019task1", "offenseval2019task2", "offenseval2019task3"]:
            return classification_report(preds, labels, target_names)
        elif task_name in ["offenseval2020task1english", "offenseval2020task2english", "offenseval2020task3english"]:
            return classification_report(preds, labels, target_names)
        elif task_name in ["offenseval2020arabic", "offenseval2020danish", "offenseval2020greek",
                           "offenseval2020turkish"]:
            return classification_report(preds, labels, target_names)
        else:
            raise KeyError(task_name)
