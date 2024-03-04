# Acronyms

## Introduction
Acronyms handing is done during inference by using preprocessing of the Hebrew sentence, which is performed before the sentence is sent for translation by the model.
The preprocessing adds the interpretation of the initials in parentheses after the initials themselves. For example, the sentence:

`"המנכ"ל ניהל את הישיבה"`

 will become after the preprocessing a sentence:

`"המנכ"ל (מנהל כללי) ניהל את הישיבה"`
## Code
### Replace acronyms with full interpretation
This is the main functionality.
**File:** [replace_acronym_with_opened_form.py](replace_acronym_with_opened_form.py)
**Input:**
- csv file containing 2 columns:
  1. (no header): acronyms
  2. (column name: `meaning`): meaning
- a Hebrew sentence, for example:
`"המנכ"ל ניהל את הישיבה"`

**Output:**
- a Hebrew sentence with full interpretation:
`"המנכ"ל (מנהל כללי) ניהל את הישיבה"`

**Comment:**
Note:
`main` also contains code processes a parquet file that contains sentences, and performs the above processing for each sentence.

### Create a dictionary of acronyms
In order to provide the described functionality, a dictionary containing acronyms and their meaning is needed. 
We support several ways to produce such a dictionary.

#### Manual dictionary creation
Of course, you can create a dictionary manually.
To save some of the manual work, the aim is to produce a dictionary that contains only relevant acronyms and fill in only the gaps.
For this we have implemented the following functionality:
1. [find_acronym.py](find_acronym.py): searches for acronyms in an external parquet file, and creates a database that contains the acronyms.
2. [retrieve_words_internal.py](retrieve_words_internal.py) (located in the utils directory) retrieves the acronyms from the files based on predefined directories structure. 
3. [extract_base_acronym.py](extract_base_acronym.py) implements a heuristic that aims to unify acronyms that contain prefixes and/or endings but their "root", or "basic version" is the same. For example: `מנכ"ל`, `המנכ"ל`, `כשהמנכ"ל`, etc. all have the "root" `מנכ"ל`.

#### Retrieving acronyms from wiktionary
[get_acronyms_from_wiktionary.py](get_acronyms_from_wiktionary.py)
Wiktionary contains a database of acronyms in different categories. We would like to use the acronyms of the IDF category.
Unfortunately, wiktionary doesn't work with `wikipediaapi`, so you have to scrape the pages, using `requests` and `BeatifulSoup`.
The `main` function contains the `base_url` - the website of Wiktionary, and the `start_url` - the first page of the of acronyms related the relevant category. Now, the code basically goes through all the values on the page, and for each value - enters the value's page and extracts the acronyms interpretation from there. 
When the transition over the values in the page is finished, it checks whether there are additional values by checking whether there is a `next page` link. If so - it moves to it. otherwise terminates.
The output is a csv file that contains the initials and their meaning.

#### Extract acronyms from a document containing the interpretation
Many times, given a document that includes acronyms, the document will contain in the first mention also the interpretation of the acronyms, near (before or after) the acronyms themselves. For example:
`'הרמטכ"ל (ראש המטה הכללי) הורה לעשות את זה'`
Therefore, if we go over  documents that deal with relevant topics, we can draw from them the acronyms and their interpretations, if they exist in the text.
It should be noted that it is necessary to treat the following cases:

The following cases are handled:
- The interpretation appears before or after the acronyms themselves, with or without brackets 
- The acronyms contain prefixes (e.g.: `הרמטכ"ל`)
- The interpretation contains prefixes (for example: `ראש המטה הכללי`)
- Each word in the interpretation can refer to one or more letters (we chose to limit it to 2 letters maximum) in the initials. For example, in the example above, `הרמטכ"ל`, the letters `מט` refer to one word - `מטה`.
**Input:** In `main` function, the `document` variable should be set to relevant sentence. for example:
`document = ''הרמטכ"ל (ראש המטה הכללי) הורה לעשות את זה'`
'
**output:**
`[('רמטכ"ל', ' ראש המטה הכללי')]`

### Sentences Decomposition
**Note:** This is an important module used in most files.

Given a sentence, [sentence_splitter.py](sentence_splitter.py) breaks the sentence into tokens, where each token is a tuple that contains one or two fields: the last field is the word itself, and the first field, if present, is the beginning. 
For example the breakdown of the word: `המל"טים` is [['ה', 'מל"טים']].

**How to run:** The code is usually called from other files, but it is also possible to run its `main`:

`sentence = 'המל"טים המריאו בחשיכה'`

**Output:**

`[['ה', 'מל"טים'], ['המריאו'], ['ב', 'חשיכה']]`

The decomposition is done using a pretrained model called `dictabert-seg`. The model is wrapped in a class called `sentenceSplitterDicta`

### Additional Files
The following files are also in use:
- [acronyms_utils.py](acronyms_utils.py)
- [regex_patterns.py](regex_patterns.py)
- [trie.py](trie.py)
