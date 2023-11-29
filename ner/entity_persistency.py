from tinydb import TinyDB, Query


class EntityPersistency:
    def __init__(self, entity_db_location):
        db = TinyDB(entity_db_location, ensure_ascii=False)
        self.db = db.table("entities")
        self.last_count = 0 # TODO: read from DB

    def increment_count(self):
        pass # TODO!!!

    def get_entity(self, entity_name: str):
        Entities = Query()
        res = self.db.search(Entities.name == entity_name)
        if not res:
            return None

        if len(res) > 1:
            print(f"ERROR: entity {entity_name} has more than one entry in the DB!!")

        return res[0]

    def get_replacement(self, entity_name: str) -> str | None:
        res = self.get_entity(entity_name)
        if not res:
            return None
        return res['replacement']

    def add_entity(self, name: str, type: str, replacement: str, source=None):
        if source is None:
            source = []
        self.db.insert({'name': name, 'type': type, 'replacement': replacement, 'source': [source]})

    def update_source(self, entity_name: str, type: str, source: str):
        res = self.get_entity(entity_name)
        if res:
            if source in res['source']:
                return # nothing to update
            if res["type"] != type:
                print(
                    f"ERROR: Entity {entity_name} has a different type in source {res['source']} than current source: {source}")

            res['source'].append(source)
            Entities = Query()
            self.db.update({'source': res['source']}, Entities.name == entity_name)

    def get_all_source_entities(self, source: str):
        Entities = Query()
        res = self.db.search(Entities.source.any([source]))
        return res


if __name__ == '__main__':
    persistency = EntityPersistency(entity_db_location=r"D:\translator\entities_test.json")
    print(persistency.get_replacement("ישראל"))
    persistency.add_entity("ישראל", "Location", "000001", "test_source")
    print(persistency.get_replacement("ישראל"))
