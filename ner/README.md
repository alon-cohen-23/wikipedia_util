## NER Augmentation.

### Introduction.

One of the most notable cases of failure is in entity translation.
Moreover, there are applications where consistency of entity translation is crucial.
All occurrences of the same person, location, or company **must** be the same across the document.
Hence, we wanted to increase the translator's performance in entity translation.

### Methods.

We want to increase performance on a specific scenario - translation of entities.
One of the most straightforward methods to do so is by adding more training samples with entities.
Getting hold of a new dataset in Hebrew is hard. Moreover, it will not be specific to entity translation.

Hence, the existing dataset is augmented.
To this end, we pass over the training dataset and tag each word for its NER tag.
NER tagging was done by first selecting a language model and fine-tuning it to the NER task
(This procedure is in the *"main"* function in *train.py*).

Having a model that could NER tag a sentence, we want to be able to replace the NER words with different words.
By that, we will create a new sentence - and augment the data.
We collected a list of first names, last names, and locations (Can be found in the *ner_list* directory).
Prefixes create a challenge.
We solve it by detecting the prefix and replacing the word only if the remaining word exists in the collected list
(This process is in *"check_ner_replacement"* in *train.py*).

This process results in an augmented dataset that can be translated and added to the training data.
Note that we can not expect better results than the translator we use (Google Translate).
A way to overcome this shortcoming is by directly replacing the name in the translation using transcription.


### Code
The code for NER (Named-entity recognition) allows identifying entities within text with their type (person, location, organization, misc.).
There are two use cases:
1. Identify and store in a (TinyDB) DB
2. Same as #1 but also replace the entity in the source dataset with a unique identifier

Notice that #2 is another step in full_flow.py (and requires the entities DB location as input) but both use cases are implemented starting in ner/entity_operations.py

Notice that the model checkpoint is global within ner/entity_operations.py