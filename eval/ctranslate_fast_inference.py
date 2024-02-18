
def ctranslate2_translate_batch(tokenizer, translator ,df_samp, col_src='he', dst_lang="eng_Latn", batch_size = 500):
    
	src_texts = []
    for index, row in df_samp.iterrows():  
        src_texts.append(row['translation'][col_src])


    translated_texts = []
    batches = [src_texts[i:i + batch_size] for i in range(0, len(src_texts), batch_size)]
    for batch in batches:
        # Tokenize the source texts using the tokenizer
        src_tokens = [tokenizer.tokenize(text, add_special_tokens=True) for text in batch]

        # Translate the batch using CTranslate2
        results = translator.translate_batch(src_tokens, target_prefix=[dst_lang]*len(src_tokens))

        # Decode the target tokens to text
        decoded_texts = [tokenizer.decode(tokenizer.convert_tokens_to_ids(result.hypotheses[0][1:])) for result in results]

        translated_texts += decoded_texts

    return translated_texts





def main(ct2_model_name_or_path, tok_name_or_path, max_samples = 4000):     
    # make predictions (translations)
    src_lang='eng_Latn' # "arb_Arab" # "heb_Hebr"
    col_src='he'
    dst_lang='heb_Hebr'
    translator = ctranslate2.Translator(ct2_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, src_lang=src_lang)
    
    # Read samples, in HF read df row['translation']['he'] and row['translation']['en']
    testset_path = Path('./data/validation.parquet')
    df_samp = pd.read_parquet(testset_path)
    
    df_samp = df_samp[:max_samples]
    translated_texts = predict(tokenizer, translator ,df_samp, col_src=col_src, dst_lang=dst_lang)
    df_samp['pred'] = translated_texts


if __name__ == "__main__":    
    df_samp = main(ct2_model_name_or_path = './output_models/nllb-200-distilled-600M_heb_eng_bidi_ctranslate2', tok_name_or_path='facebook/nllb-200-distilled-600M') 
