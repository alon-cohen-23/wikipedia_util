from tinydb import TinyDB, Query

from typing import Tuple
from utils.persistency.entry_db import EntryDB


class EntityDB(EntryDB):
    def __init__(self, name, type=None, new_name=""):
        self.name = name
        self.type = type
        self.new_name = new_name
        super(EntityDB, self).__init__()

    def _to_dict(self):
        return {'name': self.name, 'type': self.type, 'new_name': self.new_name}

    @staticmethod
    def _from_dict(d):
        e = EntityDB(d['name'])
        e.type = d['type']
        e.new_name = d['new_name']
        return e


class EntityPersistency:
    def __init__(self, entity_db_location):
        self.db = TinyDB(entity_db_location, ensure_ascii=False, encoding='utf-8')
        self.db_entities_table = self.db.table("entities")
        self.db_state_table = self.db.table("state")
        self.last_count = self.get_last_count()

    def close(self):
        self.db.close()

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
        res = self.db_entities_table.search((Entities.name == entity_name) & (Entities.type == type))
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
        entity = EntityDB(name, type)
        entity.update_source(source, 1)
        return self.upsert_entity_db(entity)

    def upsert_entity_db(self, entity: EntityDB) -> str:
        e, id = self.get_entity(entity.name, entity.type)
        if e:
            e.update_sources(entity.sources)
            self.db_entities_table.update({'sources': [s.to_dict() for s in e.sources]}, doc_ids=[id])
        else:
            entity.new_name = self.create_new_name()
            self.db_entities_table.insert(entity.to_dict())

        return entity.new_name

    def get_all_source_entities(self, source: str):
        Entities = Query()
        Sources = Query()
        res = self.db_entities_table.search(Entities.sources.any(Sources.sname == source))
        return res

    def all_entities(self):
        for e in self.db_entities_table.all():
            yield EntityDB.from_dict(e)

    def merge(self, other: 'EntityPersistency'):
        for e in other.all_entities():
            self.upsert_entity_db(e)


if __name__ == '__main__':
    persistency = EntityPersistency(entity_db_location=r"D:\translator\entities_test.json")

    print(persistency.get_all_source_entities("source1"))
    print(persistency.get_all_source_entities("source2"))

    for e in persistency.all_entities():
        print(e.name)

