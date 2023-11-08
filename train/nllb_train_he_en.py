# -*- coding: utf-8 -*-
"""NLLB_train_he_en.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DayzYm-rny-EhKMN7Om0JuRssjH23OjC
"""
"""
!pip install datasets transformers[sentencepiece]
!apt install git-lfs
!pip install numpy
!pip install transformers
!pip install sacrebleu
!pip install accelerate
!pip install tqdm
!pip install huggingface_hub
!pip install wandb
!pip install sklearn
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch
from transformers import NllbTokenizer, M2M100Tokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, load_metric, Dataset, DatasetDict, load_from_disk
from huggingface_hub import notebook_login
import loguru
import wandb

DO_TRAIN = True
DO_EVAL = True
DO_PREDICT = False

wandb.login()
# start a new wandb run to track this script
wandb.init(project="NLLB-training-project", name = "run_3_nllb_eval_only_he_en")

data_path = Path('./data')

  
def create_dataset_train_val(df_path, random_state=42, test_size=25000, 
                             max_input_length=200, max_target_length=200, train_size=-1, 
                             dataset_name = 'wikipedia_he_en_40000'):  
    df = pd.read_parquet(df_path)  
    if 'Unnamed: 0' in df.columns:  
        df = df.drop(columns=['Unnamed: 0'])  
      
    df = df[df.HE_sentences.str.len() <= max_input_length]  
    df = df[df.EN_sentences.str.len() <= max_target_length]  
    df['translation'] = df.apply(lambda row: {'en': row['EN_sentences'], 'he': row['HE_sentences']}, axis=1)    
      
    # Drop the original 'EN_sentences' and 'HE_sentences' columns    
    df = df.drop(columns=['EN_sentences', 'HE_sentences'])    
          
   
      
    # Split the dataset into a train set and a validation set (small amount)  
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)  
    
    if train_size > 0:
      train_df = train_df.iloc[:train_size]  
      
    train_dataset = Dataset.from_pandas(train_df)  
    train_dataset = train_dataset.remove_columns(['__index_level_0__'])  # Remove '__index_level_0__' feature from the datasets  
    val_dataset = Dataset.from_pandas(val_df)    
    val_dataset = val_dataset.remove_columns(['__index_level_0__'])  
    
    
    
  
    split_datasets = DatasetDict({  
        'train' : train_dataset,  
        'validation' : val_dataset,  
        })  
      
    data_folder = Path(df_path).parent  
    train_df.to_parquet(data_folder / 'train.parquet')  
    val_df.to_parquet(data_folder / 'validation.parquet')  
    split_datasets.save_to_disk(data_folder / dataset_name) 
    return split_datasets  

    


max_input_length = 200 # Max chars in source sentence
max_target_length = 200 # Max chars in target sentence







def get_output_model_name(model_checkpoint,src_lang,tgt_lang):
    # Find the index of the first '/' character
    index = model_checkpoint.find('/')
    output_name = model_checkpoint
    if index != -1:
        # Extract the prefix (facebook/) and remove it from the original string        
        output_name  = model_checkpoint[index + 1:]        
    
    output_name += f'_{src_lang.split("_")[0]}_{tgt_lang.split("_")[0]}'
    wandb.log({ "output_name": f"{output_name}" } )
    return output_name


# Language codes: https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200
model_checkpoint = "facebook/nllb-200-distilled-1.3B" # facebook/nllb-200-distilled-1.3B "facebook/m2m100_418M" # "facebook/nllb-200-distilled-600M"
    
src_lang = "heb_Hebr"
tgt_lang="eng_Latn"




output_name = get_output_model_name(model_checkpoint,src_lang,tgt_lang)


# Load the opus  dataset
# raw_datasets = load_dataset("opus100", "en-he")
#raw_datasets['train'] = raw_datasets['train'].filter(lambda example: not 'End of story.' in example['translation']['en'])
#split_datasets = raw_datasets
#del raw_datasets


# Load wikipeida dataset
split_datasets = create_dataset_train_val(df_path='./data/translated_40000_values.parquet', 
                                          max_input_length=max_input_length, 
                                          max_target_length=max_target_length,
                                          train_size=-1)




for key in split_datasets.keys():
  print (key)

metric = load_metric("sacrebleu")
#bleu_metric = load_metric("bleu")



tokenizer = NllbTokenizer.from_pretrained(model_checkpoint, src_lang=src_lang, tgt_lang=tgt_lang)


def preprocess_function(examples):

    inputs = [ex["he"] for ex in examples["translation"]]
    targets = [ex["en"] for ex in examples["translation"]]

    
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

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
    #bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)

    # connects the metrics to wandb

    # Log F1 score to WandB
    res = {"sacrebleu": scalerbleu_result["score"], "f1_score": f1}
    wandb.log(res)

    # Append the F1 score to the list for tracking
    f1_scores.append(f1)

    return res

tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)

print (tokenized_datasets)


# creates empty list to capture the f1 scores
f1_scores =[]

del split_datasets

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
batch = data_collator([tokenized_datasets["train"][i] for i in range(1, 3)])

predict_dataset=eval_dataset=tokenized_datasets["validation"]

args = Seq2SeqTrainingArguments(
    output_name,
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
    do_train=DO_TRAIN,
    do_eval=True,
    fp16=True,
    report_to="wandb"
    )

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,

)
# Access the F1 scores for each evaluation step and log them as a series
wandb.log({"f1_scores_series": wandb.Table(data=f1_scores, columns=["F1 Score"])})

# trainer.evaluate(max_length=max_target_length)
if DO_TRAIN:
    trainer.train()

if  DO_EVAL:
    print("*** Evaluate ***")

    metrics = trainer.evaluate(max_length=max_target_length, num_beams=args.generation_num_beams, metric_key_prefix="eval")    
    metrics["eval_samples"] = len(eval_dataset)

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    
if DO_PREDICT:
    print("*** Predict ***")
    predict_results = trainer.predict(
        predict_dataset, metric_key_prefix="predict", max_length=max_target_length, num_beams=args.generation_num_beams
    )
    metrics = predict_results.metrics
    
    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)

    if trainer.is_world_process_zero():
        if args.predict_with_generate:
            predictions = predict_results.predictions
            predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
            predictions = tokenizer.batch_decode(
                predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            predictions = [pred.strip() for pred in predictions]
            df = predict_dataset.to_pandas()
            df['pred'] = predictions
            df.to_parquet(Path(args.output_dir) / 'predictions.parquet')              
            
                
wandb.finish()
# trainer.evaluate(max_length=max_target_length)
# trainer.push_to_hub(tags="translation", commit_message="Training complete")

