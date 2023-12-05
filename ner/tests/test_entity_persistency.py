from ner.entity_persistency import EntityPersistency

import unittest
import tempfile


class TestEntityPersistency(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.persistency1 = EntityPersistency(entity_db_location=f"{self.temp_dir.name}/entities_test1.json")
        self.persistency2 = EntityPersistency(entity_db_location=f"{self.temp_dir.name}/entities_test2.json")

    def tearDown(self):
        self.persistency1.close()
        self.persistency2.close()
        self.temp_dir.cleanup()

    def test_upsert(self):
        self.persistency1.upsert_entity("ישראל", "Location", "source1")
        self.persistency1.upsert_entity("ישראל", "Location", "source2")
        self.persistency1.upsert_entity("ישראל", "Person", "source1")
        self.persistency1.upsert_entity("יעקב", "Person", "source1")
        self.persistency1.upsert_entity(name="بدنا_كهربا", type="Organization", source="source1")

        self.assertEqual(self.persistency1.get_new_name("ישראל", "Person"), "000001")
        self.assertEqual(self.persistency1.get_new_name("ישראל", "Location"), "000000")

        self.assertEqual(len(self.persistency1.get_all_source_entities("source1")), 4),
        self.assertEqual(len(self.persistency1.get_all_source_entities("source2")), 1),

    def test_merge(self):
        self.persistency1.upsert_entity("entity1", "Location", "source1")
        self.persistency1.upsert_entity("entity1", "Location", "source1")

        self.persistency1.upsert_entity("entity2", "Location", "source1")
        self.persistency1.upsert_entity("entity2", "Location", "source2")

        self.persistency1.upsert_entity("entity3", "Person", "source3")

        self.persistency2.upsert_entity("entity1", "Location", "source1")
        self.persistency2.upsert_entity("entity3", "Location", "source1")
        self.persistency2.upsert_entity("entity3", "Location", "source1")

        self.persistency2.merge(self.persistency1)

        self.assertEqual(self.persistency2.get_entity("entity1", "Location")[0].find_source("source1").count, 3)
        self.assertEqual(self.persistency2.get_entity("entity2", "Location")[0].find_source("source1").count, 1)
        self.assertEqual(self.persistency2.get_entity("entity2", "Location")[0].find_source("source2").count, 1)
        self.assertEqual(self.persistency2.get_entity("entity3", "Location")[0].find_source("source1").count, 2)
        self.assertEqual(self.persistency2.get_entity("entity3", "Person")[0].find_source("source3").count, 1)



if __name__ == '__main__':
    unittest.main()

