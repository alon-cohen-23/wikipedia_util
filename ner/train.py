import random
import re
from functools import partial
from typing import List

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification

from datasets import Dataset, load_metric, DatasetDict

from ner.entity_labels import EntityLabels


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
                 'B-GPE': 'B-LOC', 'S-GPE': 'B-LOC', 'I-GPE': 'I-LOC', 'E-GPE': 'I-LOC',
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


def main(out_dir=None, train_data=None, test_data=None):
    """

    Returns
    -------
    Based on trained transformer fine-tuned for the ner problem.
    """
    # Prepare constants.
    model_checkpoint = 'HeNLP/HeRo'
    model_name = model_checkpoint.split("/")[-1]
    batch_size = 16
    out_dir = out_dir if out_dir is not None else Path(__file__).parent / f"{model_name}-finetuned-ner"
    train_data = train_data if train_data is not None else r"/home/urihein/Downloads/token-multi_gold_train.bmes"
    test_data = test_data if test_data is not None else r"/home/urihein/Downloads/token-single_gold_test.bmes"
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    # Create datasets.
    datasets: DatasetDict = create_dataset(train_file=train_data, test_file=test_data)
    label_list, label_map, label2index_map, index2label_map = get_labels_maps()
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenized_datasets = datasets.map(
        partial(tokenize_and_align_labels, tokenizer=tokenizer, label_map=label_map, label2index_map=label2index_map),
        batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Load pretrained model.
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
    # Collect training arguments.
    args = TrainingArguments(
        output_dir=str(out_dir),
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=30,
        weight_decay=0.01,
        report_to=[]
    )
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


def replace_ner(sets_dict: dict[str, tuple[set, list]], ner_prediction, sentence):
    """

    Parameters
    ----------
    sets_dict: The ner lists, one time as a set (check if ner exist) the second as list (for sampling)
    ner_prediction:
    sentence

    Returns
    -------
    Returns a sentence with replaced ner
    """
    # Finding all places the sentence can be augmented.
    # List if start end location in sentence the bool is True if location.
    ner_index: List[tuple[int, int, bool]] = []
    start_person = -1
    start_location = -1
    for index, label in enumerate(ner_prediction):
        if label == EntityLabels.BeginPerson:
            if start_person >= 0:
                ner_index.append((start_person, index, False))
            start_person = index
        elif label != EntityLabels.EndPerson and start_person >= 0:
            ner_index.append((start_person, index, False))
            start_person = -1
        if label == EntityLabels.BeginLoc:
            if start_location >= 0:
                ner_index.append((start_location, index, True))
            start_location = index
        elif label != EntityLabels.EndLoc and start_location >= 0:
            ner_index.append((start_location, index, True))
            start_location = -1
    # Create the new sentence.
    new_sentence = []
    last_index = 0
    counter = 0
    for ind, t in enumerate(ner_index):
        new_sentence.extend(sentence[last_index: t[0]])
        if t[2]:
            ner_type = 'locations'
        else:
            ner_type = 'first_names'
        ner = random.sample(sets_dict[ner_type][1], 1)[0]
        current_ner = " ".join(sentence[t[0]: t[1]])
        o = re.search('^(כש|ול|[משלוב])', current_ner)
        last_index = t[1]
        if o is None:
            if current_ner == 'אבו':
                ner = current_ner + " " + ner
                last_index = t[0] + 2
                if len(ner_index) > ind + 1 and ner_index[ind + 1][0] == t[0] + 1:
                    break
            # We replace only if we identify the ner
            if current_ner not in sets_dict[ner_type][0]:
                ner = current_ner
            else:
                counter += 1
        else:
            possible_ner = current_ner[o.end():]
            # We replace only if we identify the ner
            if possible_ner == 'אבו':
                ner = current_ner + " " + ner
                last_index = t[0] + 2
                if len(ner_index) > ind + 1 and ner_index[ind + 1][0] == t[0] + 1:
                    break
            elif possible_ner in sets_dict[ner_type][0]:
                ner = current_ner[: o.end()] + ner
            elif current_ner in sets_dict[ner_type][0]:
                counter += 1
            else:
                ner = current_ner
        new_sentence.append(ner)
    new_sentence.extend(sentence[last_index:])
    return " ".join(new_sentence), counter


def test(model_checkpoint, test_file):
    """

    Parameters
    ----------
    model_checkpoint: ner model to test.
    test_file

    Returns
    -------
    Test ner model
    """
    # Load model and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
    # Prepare dataset.
    test_data = parse_file(Path(test_file))
    test_dataset = Dataset.from_list(test_data)
    label_list, label_map, label2index_map, index2label_map = get_labels_maps()
    tokenized_test_set = test_dataset.map(
        partial(tokenize_and_align_labels, tokenizer=tokenizer, label_map=label_map, label2index_map=label2index_map),
        batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    # collect argument for testing.
    metric = load_metric("seqeval")
    trainer = Trainer(model,
                      data_collator=data_collator,
                      tokenizer=tokenizer,
                      compute_metrics=partial(compute_metrics, metric=metric, label_list=label_list)
                      )
    # Test.
    predictions, labels, _ = trainer.predict(tokenized_test_set)

    # Evaluate results.
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


def align_results(tok_out, res, sentence):
    word_map = tok_out.word_ids()
    labels = [-1] * (1 + max([t for t in tok_out.word_ids() if t is not None]))
    sentence_list = []
    for i, index in enumerate(word_map):
        if index is None:
            continue
        o = tok_out.token_to_chars(i)
        if labels[index] < 0:
            sentence_list.append(sentence[o.start: o.end])
        else:
            sentence_list[-1] += sentence[o.start: o.end]

        if labels[index] < 1:
            labels[index] = res[i]

    return labels, sentence_list


def one_sentence(tokenizer, model, sentence):
    tok_out = tokenizer(sentence)
    res = model(torch.tensor([tok_out['input_ids']]),
                attention_mask=torch.tensor([tok_out['attention_mask']]))
    predicted_labels = res.logits.argmax(dim=-1).detach().numpy()[0]
    labels, sentence_list = align_results(tok_out, predicted_labels, sentence)
    return [EntityLabels(e) for e in labels], sentence_list


def load_models(model_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
    return tokenizer, model


def check_ner_replacement(
        ner_model_location=None,
        dictionaries_location=None,
        base_file_location=None,
        out_file_location=None,
        number_of_augmentations=20):
    """

    Parameters
    ----------
    number_of_augmentations: the number of augmentations for each valid sentence in the given dataset.
    ner_model_location: path to the trained ner model and tokenizer.
    dictionaries_location: path to the directory where the ner replacement list are.
    base_file_location: path to the parquet file holding all data to be augmented.
    out_file_location: path to the output directory.

    Returns
    -------
    Save the augmented data in csv file.
    """
    # Load the ner model
    ner_model_location = Path(ner_model_location) if ner_model_location is not None \
        else Path(__file__).parent / "HeRo-finetuned-ner" / "checkpoint-5500"
    tokenizer, model = load_models(ner_model_location)
    data_path = Path(dictionaries_location) if dictionaries_location is not None else Path(__file__).parent / "ner_list"
    # iter_in = ['כשאלכס ויותם הלכו לתל-אביב לפגוש את יוסי ויוני הם ראו צבי']
    # iter_in = ['אולם להבנתנו לישראל ולערב הסעודית אינטרסים משותפים גם בסוגיות אחרות, ביניהן המלחמה בדאע"ש וההתנגדות '
    #            'לאחים המוסלמים ולחמאס, כמו גם שימור היחסים המיוחדים עם ארצות הברית.']
    base_file_location = Path(
        base_file_location) if base_file_location is not None else Path(r"/home/urihein/data/voice/ner/inss")
    # Load the ner dictionaries.
    name_location_sets = {}
    for n in ['first_names', 'last_names', 'locations']:
        f = data_path / f"{n}.txt"
        s = set(f.read_text('utf-8').strip().split(', '))
        name_location_sets[n] = (s, list(s))
    # Load the dataset to be augmented.
    df = pd.read_parquet(base_file_location / "all_pages.parquet")
    iter_in = df.iloc[:, 1]
    # Create the augmented data
    res = []
    for i, sentence in enumerate(iter_in):
        if i % 100 == 0:
            print(f"{i} / {len(iter_in)}")
        labels, sentence_list = one_sentence(tokenizer, model, sentence)
        for _ in range(number_of_augmentations):
            new_sentence, number_of_replacement = replace_ner(name_location_sets, labels, sentence_list)
            if number_of_replacement > 0:
                # res.append((sentence, new_sentence, number_of_replacement))
                res.append(new_sentence)
    res_df = pd.DataFrame(res)
    print(f"Create {res_df.shape[0]} new sentences from {len(iter_in)} sentences")
    out_file_location = Path(out_file_location) if out_file_location is not None else base_file_location
    res_df.to_csv(out_file_location / "all_pages_res.csv")


if __name__ == "__main__":
    # main()
    # test(r"/home/urihein/PycharmProjects/wikipedia_util/ner/HeRo-finetuned-ner/checkpoint-5500",
    #      r"/home/urihein/Downloads/token-single_gold_test.bmes")
    check_ner_replacement()