import pandas as pd
from pathlib import Path
from nltk.tokenize import sent_tokenize


def read_file(file_name):
    with open(file_name, "rb") as f:
        data = f.read()
        # try different file encodings
        for e in ['UTF-8', 'cp1255']:
            try:
                if type(data) is bytes:
                    data = data.decode(e)
            except UnicodeDecodeError:
                pass
        lines = data.split('\n')
        sentences = []
        for line in lines:
            sentences.extend(sent_tokenize(line))
        return sentences


def read_all_files(start_dir) -> pd.DataFrame:
    data_frames = []
    for file_path in Path(start_dir).glob("*.txt"):
        file_sentences = read_file(file_path)
        data_frames.append(pd.DataFrame({'title': file_path.name, 'HE_sentences': file_sentences}))
    sentences_df = pd.concat(data_frames, ignore_index=True)
    return sentences_df


if __name__ =='__main__':
    sentences = read_file(r"d:\workspace\tr_data\INSS\data\antisemitism-education.txt")
    print(sentences)

