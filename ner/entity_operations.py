from typing import List
from functools import cached_property
from contextlib import AbstractContextManager

from ner.entity_persistency import EntityPersistency

from ner.line_info import LineInfo
from ner.entity_iterator import EntityInfo, ModelInfo, entities


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


def replace_all_lines(src_lines: List[str], src, persistency):
    res_lines = []
    for i, l in enumerate(src_lines):
        print(f"Processing line {i}/{len(src_lines)} from source {src}")
        line_info = LineInfo(l)
        with ReplacedLineBuilder(line_info) as builder:
            for entity_info in entities(model_info, line_info):
                entity_info.print_name_and_type()
                replacement_string = persistency.upsert_entity(entity_info.name, entity_info.entity_type, src)
                builder.new_entity(entity_info, replacement_string)
            res_lines.append(builder.built_line)

    return res_lines


def scan_all_lines(src_lines: List[str], src, persistency):
    for i, l in enumerate(src_lines):
        print(f"Scanning line {i}/{len(src_lines)} from source {src}: {l}")
        line_info = LineInfo(l)
        for entity_info in entities(model_info, line_info):
            # if new entity
            if persistency.get_entity(entity_info.name, entity_info.entity_type)[0] is None:
                entity_info.print_name_and_type()
            persistency.upsert_entity(entity_info.name, entity_info.entity_type, src)


if __name__ == '__main__':
    # ln1 = "מחקר במכון מוכוון מדיניות ותוצריו מיועדים לשמש את מקבלי ההחלטות במדינת ישראל ואת הציבור הרחב"
    # ln2 = "הסכמה עם ראשי הממשל האמריקאי להפסקת ההתערבות ההדדית בתהליכים פוליטיים-פנימיים בשתי המדינות."
    # ln3 = "לקדם את הרעיון של שיקום רצועת עזה, בהמשך לתהליך שהחל, לגיבוש הבנות עם החמאס לכינונה של תקופת רגיעה ממושכת."
    # ln4 = "ישראל, נלחמת בעזה."

    model_checkpoint = r"D:\translator\checkpoint-4000"
    model_info = ModelInfo(model_checkpoint)
    persistency = EntityPersistency(entity_db_location=r"D:\translator\entities.json")

    source = "wiki"
    import pandas as pd
    input_pages_file = rf"D:\workspace\tr_data\{source}/all_pages.parquet"
    df = pd.read_parquet(input_pages_file)
    lines = df["HE_sentences"].to_list()

    # lines = [ln1, ln2, ln3, ln4]
    # replaced_lines = replace_all_lines(lines, source, persistency)
    scan_all_lines(lines, source, persistency)
    print(lines)
    # print(replaced_lines)
    # df["HE_sentences"] = replaced_lines
    # df.to_parquet(rf"D:\workspace\tr_data\{source}/all_pages_er.parquet")

    # print(persistency.get_all_source_entities("source"))

