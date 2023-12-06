import regex
import pandas as pd
from tinydb import TinyDB, Query
from typing import Tuple, List
from dataset.persistency.source_db import SourceDB
from dataset.persistency.entry_db import EntryDB


class AcronymDB(EntryDB):
    def __init__(self, name):
        self.name = name
        super(AcronymDB, self).__init__()

    def _to_dict(self):
        return {'name': self.name}

    @staticmethod
    def _from_dict(d):
        return AcronymDB(d['name'])


class AcronymPersistency:
    def __init__(self, db_location):
        self.db = TinyDB(db_location, ensure_ascii=False, encoding='utf-8')
        self.db_acronyms_table = self.db.table("acronyms")

    def close(self):
        self.db.close()

    def get_acronym(self, acronym: str) -> Tuple[AcronymDB | None, int]:
        Acronyms = Query()
        res = self.db_acronyms_table.search(Acronyms.name == acronym)
        if not res:
            return None, 0

        if len(res) > 1:
            raise Exception(f"ERROR: acronym {acronym} has more than one entry in the DB!!")

        return AcronymDB.from_dict(res[0]), res[0].doc_id

    def upsert(self, name: str, source: str):
        e = AcronymDB(name)
        e.update_source(source, 1)
        self.upsert_db(e)

    def upsert_db(self, acronym: AcronymDB):
        e, id = self.get_acronym(acronym.name)
        if e:
            e.update_sources(acronym.sources)
            self.db_acronyms_table.update({'sources': [s.to_dict() for s in e.sources]}, doc_ids=[id])
        else:
            print(f"Add acronym {acronym.name} from source {source}")
            self.db_acronyms_table.insert(acronym.to_dict())


class AcronymScanner:
    def __init__(self, source, persistency: AcronymPersistency):
        self.source = source
        self.persistency = persistency

    def scan_acronyms_in_line(self, ln):
        pattern = r'\b(?P<acronym>\p{Hebrew}+\"\p{Hebrew}\p{Hebrew}*)\b'
        found = regex.findall(pattern, ln)
        if not found:
            return

        for a in found:
            self.persistency.upsert(a, self.source)

    def scan_acronyms_in_df(self, df, start_index=0):
        he_sentences = df[start_index:]["HE_sentences"].to_list()

        for i, l in enumerate(he_sentences):
            print(f"Scanning line {i + start_index + 1}/{len(he_sentences)} from source {self.source}")
            self.scan_acronyms_in_line(l)


if __name__ == "__main__":
    source = "wiki"
    #input_pages_file = rf"D:\workspace\tr_data\{source}/all_pages.parquet"
    input_pages_file = rf"D:\workspace\tr_data\{source}/translated_40000_values.parquet"
    df = pd.read_parquet(input_pages_file)

    persistency = AcronymPersistency(db_location=r"D:\translator\acronyms_test.json")
    scanner = AcronymScanner(source, persistency)
    scanner.scan_acronyms_in_df(df)

    # print("from line:")
    # line = 'המכ"ם ועוד נ"צ ועוד מטק"א'
    # scanner.scan_acronyms_in_line(line)
