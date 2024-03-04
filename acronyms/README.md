# acronyms

Contains the following functionality
1. find_acronym.py: The code will scan any of the input dataset and store acronyms in a (TinyDB) database.
2. Retrieving meaning:
   a. get_acronyms_from_wiktionary.py retrieves acronyms info from wiktionary
   b. get_acronyms_from_pdf.py retrieves acronyms info from a text file (acronyms_copied.txt) copied from a pdf source which contains domain-related acronyms 
   For now we use only the data from a since b is a bit messy
3. extract_base_acronym.py finds the 'base acronym'. This is used for 'unifying' acronyms (for example, 'המנכ"ל', 'מנכ"ל' and 'מנכ"לי' will be unified to 'מנכ"ל')

