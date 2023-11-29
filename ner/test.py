import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from wikipedia_api.ner import train
from datasets import Dataset




if __name__ == "__main__":
    main(r"/home/urihein/PycharmProjects/wikipedia_util/ner/HeRo-finetuned-ner/checkpoint-1500",
         r"/home/urihein/Downloads/token-single_gold_test.bmes")