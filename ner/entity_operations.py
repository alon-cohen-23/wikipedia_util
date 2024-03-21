from functools import cached_property
from contextlib import AbstractContextManager

from ner.entity_persistency import EntityPersistency

from ner.line_info import LineInfo
from ner.entity_iterator import EntityInfo, ModelInfo, entities

import pandas as pd


MODEL_CHECKPOINT = r"D:\translator\checkpoint-4000"


class ReplacedLineBuilder(AbstractContextManager):
    """
    Build the "replaced" line. Notice how:
    Each time an entity is found, update new line from previous substitution until the current substitution
    At the end, copy the rest of the line
    """
    def __init__(self, line_info):
        self.updated_ln = ""
        self.last_index = 0
        self.line_info = line_info

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if len(self.updated_ln) == 0:
            self.updated_ln = self.line_info.line
        else:
            self.updated_ln = self.updated_ln + self.line_info.line[self.last_index:]

        return False

    def new_entity(self, entity_info: EntityInfo, replacement_str: str):
        self.updated_ln = self.updated_ln + \
                          self.line_info.line[self.last_index:self.line_info.start(entity_info.word_start_index)] + \
                          replacement_str

        self.last_index = self.line_info.end(entity_info.word_end_index)

    @cached_property
    def built_line(self):
        if len(self.updated_ln) == 0:
            return self.line_info.line
        return self.updated_ln


class EntityOperations:
    def __init__(self, entity_json_file, source):
        self.persistency = EntityPersistency(entity_db_location=entity_json_file)
        self.model_info = ModelInfo(MODEL_CHECKPOINT)
        self.source = source

    def replace_entities_in_df(self, df: pd.DataFrame) -> pd.DataFrame:
        he_sentences = df["HE_sentences"].to_list()

        replaced_he_sentences = []
        for i, l in enumerate(he_sentences):
            print(f"Processing line {i+1}/{len(he_sentences)} from source {self.source}")
            replaced_str = self.replace_entities_in_line(l)
            replaced_he_sentences.append(replaced_str)

        res_df = df.copy()
        res_df["HE_sentences"] = replaced_he_sentences
        return res_df

    def replace_entities_in_line(self, ln: str) -> str:
        line_info = LineInfo(ln)
        with ReplacedLineBuilder(line_info) as builder:
            for entity_info in entities(self.model_info, line_info):
                entity_info.print_name_and_type()
                replacement_string = self.persistency.upsert_entity(entity_info.name, entity_info.entity_type,
                                                                    self.source)
                builder.new_entity(entity_info, replacement_string)
            return builder.built_line

    def scan_entities_in_df(self, df: pd.DataFrame, start_index=0):
        he_sentences = df[start_index:]["HE_sentences"].to_list()

        for i, l in enumerate(he_sentences):
            print(f"Scanning line {i+start_index+1}/{len(he_sentences)} from source {self.source}")
            self.scan_entities_in_line(l)

    def scan_entities_in_line(self, ln: str):
        line_info = LineInfo(ln)
        for entity_info in entities(self.model_info, line_info):
            if self.persistency.get_entity(entity_info.name, entity_info.entity_type)[0] is None:
                entity_info.print_name_and_type()
            self.persistency.upsert_entity(entity_info.name, entity_info.entity_type, self.source)


def replace_entities(df, source, entity_db_location):
    en_op = EntityOperations(entity_db_location, source)
    return en_op.replace_entities_in_df(df)


if __name__ == '__main__':
    ln1 = "מחקר במכון מוכוון מדיניות ותוצריו מיועדים לשמש את מקבלי ההחלטות במדינת ישראל ואת הציבור הרחב"
    ln2 = "הסכמה עם ראשי הממשל האמריקאי להפסקת ההתערבות ההדדית בתהליכים פוליטיים-פנימיים בשתי המדינות."
    ln3 = "לקדם את הרעיון של שיקום רצועת עזה, בהמשך לתהליך שהחל, לגיבוש הבנות עם החמאס לכינונה של תקופת רגיעה ממושכת."
    ln4 = "ישראל, נלחמת בעזה."
    lines = [ln1, ln2, ln3, ln4]

    entity_db_location = r"/home/urihein/Downloads/entities_fauda_teheran_inss.json"
    source = "inss"
    en_op = EntityOperations(entity_db_location, source)

    input_pages_file = rf"D:\workspace\tr_data\{source}/all_pages.parquet"
    df = pd.read_parquet(input_pages_file)

    en_op.scan_entities_in_df(df)
    #en_op.scan_entities_in_line(ln1)
    #en_op.scan_entities_in_line(ln2)

    # res_df = en_op.replace_entities_in_df()
    # res_df.to_parquet(rf"D:\workspace\tr_data\{source}/all_pages_er.parquet")

    # print(persistency.get_all_source_entities("source"))hy
