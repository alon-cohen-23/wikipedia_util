# wikipedia_util
## Prepare training data
All dependencies for preparing the datasets is stored in dataset/requirements.txt

After installation, make sure to run the following Python code from a Python terminal *once* before running:
```
import nltk
nltk.download('punkt')
```
### wikipedia
TODO
### Text files
We assume that the input is a collection of text files. See dataset/full_flow.py for documentation. 
### Acronyms
The code within the acronym directory will scan any of the input dataset and store acronyms in a (TinyDB) database.
## NER
The code for NER (Named-entity recognition) allows identifying entities within text with their type (person, location, organization, misc.).
There are two use cases:
1. Identify and store in a (TinyDB) DB
2. Same as #1 but also replace the entity in the source dataset with a unique identifier

Notice that #2 is another step in full_flow.py (and requires the entities DB location as input) but both use cases are implemented starting in ner/entity_operations.py

Notice that the model checkpoint is global within ner/entity_operations.py
## Train and Evaluate Model
### wandb
Create user.
When running, put the API key in the WANDB_API_KEY environment variable
## Translate
TODO