from ner.entity_labels import EntityLabels
from typing import Iterator, Tuple, List


def label_entity_iterator(labels: List[int]) -> Iterator[Tuple[List[int], str]]:
    """
    Iterator that receives NER model labels and returns a tuple consisting of the entity indices and it's type
    Parameters
    ----------
    labels(array of integers): labels within the word list

    Returns
    -------
    Iterator
    """

    last_entity_begin = EntityLabels.NoEntity
    last_entity_indices = []

    if labels is None:
        raise StopIteration

    for i, e in enumerate(labels):
        entity_marker = EntityLabels(e)

        if entity_marker.no_entity() or entity_marker.is_begin():
            # check if previous entity completed
            if len(last_entity_indices) != 0:
                yield last_entity_indices, last_entity_begin.to_type()
                last_entity_indices.clear()
            last_entity_begin = entity_marker

        if entity_marker.is_begin():
            last_entity_indices.append(i)
        elif entity_marker.is_end():
            if EntityLabels.begin_end_match(last_entity_begin, entity_marker):
                last_entity_indices.append(i)

    if len(last_entity_indices) != 0:
        yield last_entity_indices, last_entity_begin.to_type()
        last_entity_indices.clear()


if __name__ == '__main__':
    ln1 = "מחקר במכון מוכוון מדיניות ותוצריו מיועדים לשמש את מקבלי ההחלטות במדינת ישראל ואת הציבור הרחב"
    labels1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0]
    ln2 = "לקדם את הרעיון של שיקום רצועת עזה, בהמשך לתהליך שהחל, לגיבוש הבנות עם החמאס לכינונה של תקופת רגיעה ממושכת."
    labels2 = [0, 0, 0, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0]

    print(labels2)
    for indices, entity_type in label_entity_iterator(labels2):
        print(f"{entity_type}: {indices}")
