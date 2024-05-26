from typing import Dict, List
from enum import Enum, IntEnum


class EntityLabels(IntEnum):
    """
    This enum maps the NER model labels to a human-readable naming + supporting functions.
    Notice that besides NoEntity, the rest are all pairs of entity types
    """
    NoEntity = 0,
    BeginPerson = 1,
    EndPerson = 2,
    BeginOrg = 3,
    EndOrg = 4,
    BeginLoc = 5,
    EndLoc = 6,
    BeginMisc = 7,
    EndMisc = 8

    def to_type(self):
        """
        Convert the enum to the corresponding type string
        Returns
        -------
        String
        """
        for name, p in EntityLabels.pairs().items():
            if self in p:
                return name
        return "NoEntity"

    def no_entity(self) -> bool:
        return self == EntityLabels.NoEntity

    def is_begin(self) -> bool:
        return self in [EntityLabels.BeginPerson, EntityLabels.BeginOrg, EntityLabels.BeginLoc, EntityLabels.BeginMisc]

    def is_end(self) -> bool:
        return self in [EntityLabels.EndPerson, EntityLabels.EndOrg, EntityLabels.EndLoc, EntityLabels.EndMisc]

    @classmethod
    def pairs(cls) -> Dict[str, List[IntEnum]]:
        pairs_dict = {
            "Person": [EntityLabels.BeginPerson, EntityLabels.EndPerson],
            "Organization": [EntityLabels.BeginOrg, EntityLabels.EndOrg],
            "Location": [EntityLabels.BeginLoc, EntityLabels.EndLoc],
            "Misc": [EntityLabels.BeginMisc, EntityLabels.EndMisc]
        }
        return pairs_dict

    @classmethod
    def begin_end_match(cls, begin, end) -> bool:
        """
        Check if the (begin, end) pair belong to the same entity type
        """
        return [begin, end] in cls.pairs().values()


if __name__ == '__main__':
    print(EntityLabels.BeginOrg.is_begin())
    print(EntityLabels.BeginOrg.is_end())
    print(EntityLabels.EndPerson.is_begin())
    print(EntityLabels.EndPerson.is_end())
    print(EntityLabels.BeginLoc.to_type())
