from tinydb import TinyDB, Query

from typing import List, Tuple


class SourceDB:
    def __init__(self, sname, count=0):
        self.sname = sname
        self.count = count

    def to_dict(self):
        return {'sname': self.name, 'count': self.count}

    @staticmethod
    def from_dict(d):
        return SourceDB(d['sname'], d['count'])


class EntityDB:
    def __init__(self, name, type=None, new_name=""):
        self.name = name
        self.type = type
        self.new_name = new_name
        self.sources: List[SourceDB] = []

    def update_source(self, source: str):
        source_loc = -1
        for i, s in enumerate(self.sources):
            if s.name == source:
                source_loc = i
        if source_loc >= 0:
            self.sources[source_loc].count += 1
        else:
            self.sources.append(SourceDB(source, 1))

    def to_dict(self):
        d = {'name': self.name, 'type': self.type, 'new_name': self.new_name, 'sources': []}

        for s in self.sources:
            d['sources'].append(s.to_dict())

        return d

    @staticmethod
    def from_dict(d):
        e = EntityDB(d['name'])
        e.type = d['type']
        e.new_name = d['new_name']
        e.sources = []
        for s in d['sources']:
            e.sources.append(SourceDB.from_dict(s))
        return e


class EntityPersistency:
    def __init__(self, entity_db_location):
        db = TinyDB(entity_db_location, ensure_ascii=False, encoding='utf-8')
        self.db = db.table("entities")
        self.db_state_table = db.table("state")
        self.last_count = self.get_last_count()

    def get_last_count(self):
        res = self.db_state_table.all()
        if res:
            return res[0]["last_count"]

        self.db_state_table.insert({"last_count": 0})
        return 0

    def increment_count(self):
        self.last_count += 1
        self.db_state_table.update({'last_count': self.last_count})

    def get_entity(self, entity_name: str, type: str) -> Tuple[EntityDB | None, int]:
        Entities = Query()
        res = self.db.search((Entities.name == entity_name) & (Entities.type == type))
        if not res:
            return None, 0

        if len(res) > 1:
            raise Exception(f"ERROR: entity {entity_name} has more than one entry in the DB!!")

        return EntityDB.from_dict(res[0]), res[0].doc_id

    def get_new_name(self, entity_name: str, type: str) -> str | None:
        res, id = self.get_entity(entity_name, type)
        if not res:
            return None
        return res.new_name

    def create_new_name(self):
        new_name = f"{self.last_count:06d}"
        self.increment_count()
        return new_name

    def upsert_entity(self, name: str, type: str, source: str) -> str:
        e, id = self.get_entity(name, type)
        if e:
            e.update_source(source)
            self.db.update({'sources': [s.to_dict() for s in e.sources]}, doc_ids=[id])
        else:
            e = EntityDB(name, type, self.create_new_name())
            e.sources.append(SourceDB(source, 1))
            self.db.insert(e.to_dict())
        return e.new_name

    def get_all_source_entities(self, source: str):
        Entities = Query()
        Sources = Query()
        res = self.db.search(Entities.sources.any(Sources.name == source))
        return res


if __name__ == '__main__':
    persistency = EntityPersistency(entity_db_location=r"D:\translator\entities_test.json")
    persistency.upsert_entity("ישראל", "Location", "source1")
    persistency.upsert_entity("ישראל", "Location", "source2")
    persistency.upsert_entity("ישראל", "Person", "source1")
    persistency.upsert_entity("יעקב", "Person", "source1")
    persistency.upsert_entity(name="بدنا_كهربا", type="Organization", source="source1")

    print(persistency.get_new_name("ישראל", "Person"))
    print(persistency.get_new_name("ישראל", "Location"))

    print(persistency.get_all_source_entities("source1"))
    print(persistency.get_all_source_entities("source2"))