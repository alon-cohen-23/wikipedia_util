from ner.entity_predication import predict_entities, initialize_model_and_tokenizer
from ner.label_entity_iterator import label_entity_iterator

from ner.line_info import LineInfo

from typing import Iterator, List
from functools import cached_property


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
        return self.line_info.get_name(self.word_start_index, self.word_end_index)
        # return " ".join(self.line_info.words[self.word_start_index:self.word_end_index])

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
