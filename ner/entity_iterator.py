from ner.entity_predication import predict_entities, initialize_model_and_tokenizer
from ner.label_entity_iterator import label_entity_iterator

from ner.line_info import LineInfo

from typing import Iterator, List
from functools import cached_property


def is_nikud(c):
    # notepad++ regular expression search: [\x{0591}-\x{05BD}\x{05BF}-\x{05C2}\x{05C4}-\x{05C7}]
    ord_c = ord(c)
    return 0x0591 <= ord_c <= 0x5BD or 0x5BF <= ord_c <= 0x5C2 or 0x5C4 <= ord_c <= 0x5C7


class EntityInfo:
    def __init__(self, indices: List[int], entity_type: str, line_info: LineInfo):
        self.word_indices = indices
        self.entity_type = entity_type
        line_info.strip_punctuation(self.word_start_index, self.word_end_index)
        self.line_info = line_info

    @cached_property
    def word_start_index(self):
        return self.word_indices[0]

    @cached_property
    def word_end_index(self):
        return self.word_indices[-1] + 1

    @cached_property
    def name(self):
        entity_name = self.line_info.get_name(self.word_start_index, self.word_end_index)
        return "".join(filter(lambda c: not is_nikud(c), entity_name))

    def print_name_and_type(self):
        print(f"Found entity = {self.name}, type = {self.entity_type}")


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


def entities(ner_model_info: ModelInfo, line_info: LineInfo) -> Iterator[EntityInfo]:
    """
    Iterator that returns an EntityInfo class for each entity returned by label_entity_iterator
    """
    labels = ner_model_info.predict(line_info)
    for (indices, entity_type) in label_entity_iterator(labels):
        yield EntityInfo(indices, entity_type, line_info)
