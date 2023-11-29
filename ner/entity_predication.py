from typing import List
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification


def pre_process(row):
    return row


def post_process(res):
    return res


def predict_row(row: List[str], tokenizer, model, gpu=False):
    # each word in a different cell in the list row.
    row = pre_process(row)
    tokenized_inputs = tokenizer(row, truncation=True, is_split_into_words=True)
    if gpu:
        res = model(torch.tensor([tokenized_inputs['input_ids']]).cuda()).logits.cpu().detach().numpy()
    else:
        res = model(torch.tensor([tokenized_inputs['input_ids']])).logits.detach().numpy()
    res = post_process(np.argmax(res, axis=2))
    return res[0], tokenized_inputs.word_ids(batch_index=0)


def initialize_model_and_tokenizer(model_checkpoint, gpu=False):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
    if not gpu:
        model = model.to('cpu')

    return tokenizer, model


def predict_entities(model, tokenizer, words):
    res, word_map = predict_row(words, tokenizer, model)
    labels = [None] * len(words)
    for i, index in enumerate(word_map):
        if index is None:
            continue
        labels[index] = res[i] if labels[index] is None or labels[index] < 1 else labels[index]

    return labels


if __name__ == '__main__':
    model_checkpoint = r"D:\translator\checkpoint-4000"
    tokenizer, model = initialize_model_and_tokenizer(model_checkpoint)
    ln1 = "מחקר במכון מוכוון מדיניות ותוצריו מיועדים לשמש את מקבלי ההחלטות במדינת ישראל ואת הציבור הרחב"
    ln2 = "הסכמה עם ראשי הממשל האמריקאי להפסקת ההתערבות ההדדית בתהליכים פוליטיים-פנימיים בשתי המדינות."
    ln3 = "לקדם את הרעיון של שיקום רצועת עזה, בהמשך לתהליך שהחל, לגיבוש הבנות עם החמאס לכינונה של תקופת רגיעה ממושכת."
    #words = ln1.split(" ")
    #words = ln2.split(" ")
    words = ln3.split(" ")
    labels = predict_entities(model, tokenizer, words)
    print(labels)
