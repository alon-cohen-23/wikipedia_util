from functools import partial
from typing import List

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification

from datasets import Dataset, load_metric, DatasetDict


# https://github.com/huggingface/notebooks/blob/main/examples/token_classification.ipynb
def compute_metrics(p, label_list, metric):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def parse_file(data_file: Path):
    with data_file.open():
        txt = data_file.read_text()
        rows = txt.split("\n\n")
        data = []
        for row in rows:
            row_data = [r.split(' ') for r in row.split("\n")]
            d = {t[1]: t[0] for t in list(zip(zip(*row_data), ['tokens', 'labels']))}
            if 'labels' not in d:
                continue
            data.append(d)
    return data


def create_dataset(train_file, test_file):
    train_data = parse_file(Path(train_file))
    test_data = parse_file(Path(test_file))
    validation_size = int(len(train_data) * 0.1)
    validation_index = np.random.choice(len(train_data), validation_size)
    validation_data = [train_data[i] for i in validation_index]
    train_data = [t for i, t in enumerate(train_data) if i not in validation_index]
    train_dataset = Dataset.from_list(train_data)
    validation_dataset = Dataset.from_list(validation_data)
    test_dataset = Dataset.from_list(test_data)
    return DatasetDict(train=train_dataset, validation=validation_dataset, test=test_dataset)


def get_labels_maps():
    label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    label2index_map = {l: i for i, l in enumerate(label_list)}
    index2label_map = {v: k for k, v in label2index_map.items()}
    label_map = {'O': 'O',
                 'B-ORG': 'B-ORG', 'S-ORG': 'B-ORG', 'I-ORG': 'I-ORG', 'E-ORG': 'I-ORG',
                 'B-PER': 'B-PER', 'S-PER': 'B-PER', 'I-PER': 'B-PER', 'E-PER': 'I-PER',
                 'B-GPE': 'B-ORG', 'S-GPE': 'B-ORG', 'I-GPE': 'I-ORG', 'E-GPE': 'I-ORG',
                 'B-LOC': 'B-LOC', 'S-LOC': 'B-LOC', 'I-LOC': 'I-LOC', 'E-LOC': 'I-LOC',
                 'B-FAC': 'B-MISC', 'S-FAC': 'B-MISC', 'I-FAC': 'I-MISC', 'E-FAC': 'I-MISC',
                 'B-DUC': 'I-MISC', 'S-DUC': 'B-MISC', 'I-DUC': 'I-MISC', 'E-DUC': 'I-MISC',
                 'B-EVE': 'B-MISC', 'S-EVE': 'B-MISC', 'I-EVE': 'I-MISC', 'E-EVE': 'I-MISC',
                 'B-WOA': 'B-MISC', 'S-WOA': 'B-MISC', 'I-WOA': 'I-MISC', 'E-WOA': 'I-MISC',
                 'B-ANG': 'B-MISC', 'S-ANG': 'B-MISC', 'I-ANG': 'I-MISC', 'E-ANG': 'I-MISC',
                 }
    return label_list, label_map, label2index_map, index2label_map


def tokenize_and_align_labels(examples, tokenizer, label_map, label2index_map, label_all_tokens=False):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            else:
                current_label = label2index_map[label_map.get(label[word_idx], "O")]
                if word_idx != previous_word_idx:
                    label_ids.append(current_label)
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(current_label if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def main():
    task = "ner"  # Should be one of "ner", "pos" or "chunk"
    model_checkpoint = 'HeNLP/HeRo'
    batch_size = 16
    out_dir = Path(r"/home/urihein/data/voice/ner/")
    out_dir.mkdir(exist_ok=True, parents=True)

    datasets: DatasetDict = create_dataset(train_file=r"/home/urihein/Downloads/token-multi_gold_train.bmes",
                                           test_file=r"/home/urihein/Downloads/token-single_gold_test.bmes")
    label_list, label_map, label2index_map, index2label_map = get_labels_maps()
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenized_datasets = datasets.map(
        partial(tokenize_and_align_labels, tokenizer=tokenizer, label_map=label_map, label2index_map=label2index_map),
        batched=True)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
    model_name = model_checkpoint.split("/")[-1]
    args = TrainingArguments(
        # output_dir=str(out_dir),
        f"{model_name}-finetuned-{task}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=15,
        weight_decay=0.01,
        report_to=[]
    )
    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = load_metric("seqeval")
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, metric=metric, label_list=label_list)
    )
    trainer.train()


def test(model_checkpoint, test_file):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
    test_data = parse_file(Path(test_file))
    test_dataset = Dataset.from_list(test_data)
    label_list, label_map, label2index_map, index2label_map = get_labels_maps()
    tokenized_test_set = test_dataset.map(
        partial(tokenize_and_align_labels, tokenizer=tokenizer, label_map=label_map, label2index_map=label2index_map),
        batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = load_metric("seqeval")

    trainer = Trainer(model,
                      data_collator=data_collator,
                      tokenizer=tokenizer,
                      compute_metrics=partial(compute_metrics, metric=metric, label_list=label_list)
                      )
    predictions, labels, _ = trainer.predict(tokenized_test_set)
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    print(results)



if __name__ == "__main__":
    # main()
    test(r"/home/urihein/PycharmProjects/wikipedia_util/ner/HeRo-finetuned-ner/checkpoint-4000",
         r"/home/urihein/Downloads/token-single_gold_test.bmes")