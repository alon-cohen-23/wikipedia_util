# Persistency

The persistency directory includes code for tracking entities in a TinyDB database (tiny document DB).
The database consists of a list of entities, each uniquely identified (id + name). 
For each entity, store the list of data sources (where this entity was found).

This code is used:
- For building the acronyms database (AcronymDB inherits from EntryDB)
- For building the entities database (EntityDB inherits from EntryDB)