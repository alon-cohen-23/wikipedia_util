import traceback

STEPS = 5

from datetime import datetime
import builtins


def print(*args, **kwargs):
    builtins.print(f"{datetime.now()} - {args[0]}")


def main(initial_checkpoint_model, wiki_percentage):
    try:
        print(f"STEP 1 / {STEPS} : Load Dependencies")

        from pathlib import Path
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score
        import torch
        from transformers import NllbTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, \
            Seq2SeqTrainer
        from datasets import load_dataset, load_metric, Dataset, DatasetDict, load_from_disk, concatenate_datasets
        import os

        OUTPUT_DIRECTORY = os.path.join(os.getcwd(), 'output')

        print(f"STEP 2 / {STEPS} : Organize the data")

        from datasets import concatenate_datasets, load_dataset

        tokenized_datasets_teheran = load_from_disk('/home/azureuser/translator/tr_data/teheran/prepared_dataset')
        tokenized_datasets_fauda = load_from_disk('/home/azureuser/translator/tr_data/fauda/prepared_dataset')
        tokenized_datasets_inss = load_from_disk('/home/azureuser/translator/tr_data/inss/prepared_dataset')
        tokenized_datasets_wiki = load_from_disk('/home/azureuser/translator/tr_data/wiki/prepared_dataset')

        train_dataset = concatenate_datasets([
            tokenized_datasets_teheran['train'],
            tokenized_datasets_fauda['train'],
            tokenized_datasets_inss['train'],
            tokenized_datasets_wiki['train'].select(
                range(int(tokenized_datasets_wiki['train'].num_rows * wiki_percentage)))
        ])

        validation_dataset = concatenate_datasets([
            tokenized_datasets_teheran['validation'],
            tokenized_datasets_fauda['validation'],
            tokenized_datasets_inss['validation'],
            tokenized_datasets_wiki['validation'].select(
                range(int(tokenized_datasets_wiki['validation'].num_rows * wiki_percentage)))
        ])

        tokenized_datasets = DatasetDict({
            'train': train_dataset,
            'validation': validation_dataset,
        })

        tokenized_datasets.save_to_disk('./tr_data/tokenized_dataset_used')

        print(f"STEP 3 / {STEPS} :Load the Model")

        src_lang = "heb_Hebr"
        tgt_lang = "eng_Latn"

        model_checkpoint = initial_checkpoint_model
        # Using local version
        # model_checkpoint = "/data2/translation/nllb/nllb-200-distilled-600M-he-en/checkpoint-124998/"

        tokenizer = NllbTokenizer.from_pretrained(model_checkpoint, src_lang=src_lang, tgt_lang=tgt_lang)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        print(f"STEP 4 / {STEPS} : Load The metric")

        metric = load_metric("sacrebleu")

        f1_scores = []

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # In case the model returns more than the prediction logits
            if isinstance(preds, tuple):
                preds = preds[0]

            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

            # Replace -100s in the labels as we can't decode them
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Some simple post-processing
            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = [[label.strip()] for label in decoded_labels]

            # calculate the f1 score
            f1 = f1_score(decoded_labels, decoded_preds, average='weighted')

            # calculate scalerblue results
            scalerbleu_result = metric.compute(predictions=decoded_preds, references=decoded_labels)
            # bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)

            # connects the metrics to wandb

            # Log F1 score to WandB
            res = {"sacrebleu": scalerbleu_result["score"], "f1_score": f1}

            # Append the F1 score to the list for tracking
            f1_scores.append(f1)

            return res

        print(f"STEP 5 / {STEPS} : Setup Train ")

        args = Seq2SeqTrainingArguments(
            OUTPUT_DIRECTORY,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=2,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            eval_accumulation_steps=3,
            predict_with_generate=True,
            push_to_hub=False,
            do_train=True,
            do_eval=True,
            fp16=True
        )

        trainer = Seq2SeqTrainer(
            model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,

        )

        trainer.train()
        print(f"STEP 6 / {STEPS} : Train is done")

        model_to_save = f'./models/{initial_checkpoint_model.replace("/", "_")}-finetune-all-data_wiki_{wiki_percentage}'
        trainer.save_model(model_to_save)

        print(f"STEP 7 / {STEPS} : Model is saved under {model_to_save} ")
    except Exception as e:
        print(traceback.print_exc())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        prog='trainer',
        description='Train a model on all the data',
        epilog='Fee free to contact me')

    parser.add_argument('--model', type=str, default="facebook/nllb-200-distilled-1.3B")
    parser.add_argument('--wiki', type=float, default=0.5)

    args = parser.parse_args()
    print(args.model)
    print(args.wiki)
    main(initial_checkpoint_model=args.model, wiki_percentage=args.wiki)
