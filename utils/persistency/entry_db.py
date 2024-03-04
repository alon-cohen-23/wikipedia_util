from typing import List
from utils.persistency.source_db import SourceDB


class EntryDB:
    def __init__(self):
        self.sources: List[SourceDB] = []

    def find_source_index(self, source) -> int:
        source_loc = -1
        for i, s in enumerate(self.sources):
            if s.sname == source:
                source_loc = i

        return source_loc

    def find_source(self, source) -> SourceDB | None:
        source_loc = self.find_source_index(source)
        if source_loc >= 0:
            return self.sources[source_loc]
        return None

    def update_source(self, source: str, count=1):
        source_loc = self.find_source_index(source)

        if source_loc >= 0:
            self.sources[source_loc].count += count
        else:
            self.sources.append(SourceDB(source, count))

    def update_sources(self, sources: List[SourceDB]):
        for s in sources:
            self.update_source(s.sname, s.count)

    def _to_dict(self):
        return {}

    def to_dict(self):
        d = self._to_dict()
        d['sources'] = []
        for s in self.sources:
            d['sources'].append(s.to_dict())

        return d

    @staticmethod
    def _from_dict(d):
        return object

    @classmethod
    def from_dict(cls, d):
        e = cls._from_dict(d)
        e.sources = []
        for s in d['sources']:
            e.sources.append(SourceDB.from_dict(s))
        return e
