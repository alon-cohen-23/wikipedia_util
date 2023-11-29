from typing import Iterator, List
from functools import lru_cache, cached_property
from contextlib import AbstractContextManager


import nltk
from nltk.tokenize import NLTKWordTokenizer

from ner.entity_predication import predict_entities, initialize_model_and_tokenizer
from ner.entity_iterator import label_entity_iterator
from ner.entity_persistency import EntityPersistency


class LineInfo:
    def __init__(self, line):
        self.line = line
        self.words = nltk.word_tokenize(line)
        self.word_spans = list(NLTKWordTokenizer().span_tokenize(line))

    @lru_cache
    def start(self, word_start_index):
        return self.word_spans[word_start_index][0]

    @lru_cache
    def end(self, word_end_index):
        return self.word_spans[word_end_index - 1][1]


class EntityInfo:
    def __init__(self, indices: List[int], entity_type: str, line_info: LineInfo):
        self.word_indices = indices
        self.entity_type = entity_type
        self.line_info = line_info

    @cached_property
    def word_start_index(self):
        return self.word_indices[0]

    @cached_property
    def word_end_index(self):
        return self.word_indices[-1] + 1

    @cached_property
    def name(self):
        return " ".join(self.line_info.words[self.word_start_index:self.word_end_index])

    def print_name_and_type(self):
        print(f"Found entity = {self.name}, type = {self.entity_type}")

    def track(self, persistency: EntityPersistency, source: str):
        """
        Search for replacement string matching this entity.
        - If not found, create one and update DB.
        - If found, update source (add to list if doesn't exist)
        Parameters
        ----------
        persistency (EntityPersistency): class responsible for communicating with the DB
        source (str): entity source (e.g. "wiki")

        Returns
        -------
        String that can be used to replace the entity in the sentence
        """
        replacement_string = persistency.get_replacement(self.name)
        if replacement_string is None:
            replacement_string = f"{persistency.last_count:06d}"
            persistency.increment_count()
            persistency.add_entity(self.name, self.entity_type, replacement_string, source)
        else:
            persistency.update_source(self.name, self.entity_type, source)
        return replacement_string


class ModelInfo:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint

    @cached_property
    def tokenizer_model(self):
        return initialize_model_and_tokenizer(self.checkpoint)

    def predict(self, line_info: LineInfo):
        t, m = self.tokenizer_model
        labels = predict_entities(m, t, words=line_info.words)
        return labels


def entities(model_info: ModelInfo, line_info: LineInfo) -> Iterator[EntityInfo]:
    """
    Iterator that returns an EntityInfo class for each entity returned by label_entity_iterator
    """
    labels = model_info.predict(line_info)
    for (indices, entity_type) in label_entity_iterator(labels):
        entity_info = EntityInfo(indices, entity_type, line_info)
        entity_info.print_name_and_type()

        yield entity_info


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


if __name__ == '__main__':
    ln1 = "מחקר במכון מוכוון מדיניות ותוצריו מיועדים לשמש את מקבלי ההחלטות במדינת ישראל ואת הציבור הרחב"
    ln2 = "הסכמה עם ראשי הממשל האמריקאי להפסקת ההתערבות ההדדית בתהליכים פוליטיים-פנימיים בשתי המדינות."
    ln3 = "לקדם את הרעיון של שיקום רצועת עזה, בהמשך לתהליך שהחל, לגיבוש הבנות עם החמאס לכינונה של תקופת רגיעה ממושכת."
    ln4 = "ישראל, נלחמת בעזה."

    model_checkpoint = r"D:\translator\checkpoint-4000"
    model_info = ModelInfo(model_checkpoint)
    persistency = EntityPersistency(entity_db_location=r"D:\translator\entities.json")

    lines = [ln1, ln2, ln3, ln4]
    replaced_lines = []
    # lines = [ln3, ln4]
    for l in lines:
        line_info = LineInfo(l)
        for entity_info in entities(model_info, line_info):
            replacement_string = entity_info.track(persistency, "inss")
            with ReplacedLineBuilder(line_info) as builder:
                builder.new_entity(entity_info, replacement_string)
        replaced_lines.append(builder.updated_ln)

    print(lines)
    print(replaced_lines)
    # print(persistency.get_all_source_entities("inss"))
    # print(persistency.get_all_source_entities("teheran"))
    # print(persistency.get_all_source_entities("wiki"))
    #
