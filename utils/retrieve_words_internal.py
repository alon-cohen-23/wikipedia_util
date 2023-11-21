from collections import Counter
from pathlib import Path
import json, os, random, string, regex, argparse, pickle
import pandas as pd

# פונקציה שמקבלת טקסט. תפעיל לולאה על json שמכיל מערך של מילונים הנקרא Results ובכל מילון יש מפתח Content
# ועוד לולאה חיצונית שעוברת על פולדר עם הרבה json files. בכל אחד מהם מערך Results שבכל אחד מאברי המערך מילון עם Content

# for testing
def create_random_json(filename = None):
    letters = string.ascii_lowercase
    n_elements = random.randint(3, 10)
    results = []
    for i in range(n_elements):
        dict_len = random.randint(3, 10)
        random_location = random.randint(0, dict_len-1)
        d = {}
        for j in range(dict_len):
            if j==random_location:
                # d['content'] = f'This is my content: {j}'
                d['content'] = 'תנו לצה"ל והשב"כ לנצח'
            else:
                random_key = ''.join(random.choice(letters) for i in range(10))
                d[random_key] = ''.join(random.choice(letters) for i in range(10))
        results.append(d)
    json_object = json.dumps({'Results': results})
    with open(filename, "w") as outfile:
        outfile.write(json_object)

def create_json_dir(json_dir):
    Path(json_dir).mkdir(parents=True, exist_ok=True)
    n_files = random.randint(3, 10)
    for i in range(n_files):
        create_random_json(os.path.join(json_dir, f'{i}.json'))


def retrieve_content_from_json(filename):
    contents = []
    with open(filename) as f:
        data = json.load(f)
    try:
        results = data['Results']
        for res in results:
            contents.append(res['content'])
    except:
        print(f'Incorrect file structure {filename}')
    return contents

def retrieve_json_dir(json_dir):
    content_in_dir = []
    for filename in os.listdir(json_dir):
        content_in_dir+= retrieve_content_from_json(os.path.join(json_dir, filename))
    return content_in_dir

def count_acronyms(l_txts):
    counter = Counter()
    pattern = "\p{Hebrew}+\"\p{Hebrew}\p{Hebrew}*"
    for txt in l_txts:
        found = regex.findall(pattern, txt)
        if found:
            counter.update(found)
    return counter

def count_words(l_txts):
    counter = Counter()
    for txt in l_txts:
        words = txt.split()
        if len(words)>0:
            counter.update(words)
    return counter

def store_common(count: Counter, n: int, output_file: str):
    most_common_items = count.most_common(n)
    pd.DataFrame(most_common_items, columns=["item", "count"]).to_csv(output_file, encoding = "ISO-8859-8")

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        prog='retrieveInsideData',
        description='Retrieve data from internal json files and count words/acronyms')

    parser.add_argument('-j', '--json_dir', default='json_dir')
    parser.add_argument('-c', '--create', action='store_true', help='Create dir with jsons. Used for testing')
    parser.add_argument('-a', '--n_acronyms', type=int, default=0,
                        help='Number of most common acronyms to retrieve (0 - no acronyms count)')
    parser.add_argument('-w', '--n_words', type=int, default=0,
                        help='Number of most common words to retrieve (0 - no words count)')

    args = parser.parse_args()

    if args.create:
        create_json_dir(args.json_dir)

    if args.n_acronyms or args.n_words:
        content_in_dir = retrieve_json_dir(args.json_dir)
        if args.n_acronyms:
            counts = count_acronyms(content_in_dir)
            store_common(counts, args.n_acronyms, "n_acronyms.csv")
        if args.n_words:
            counts = count_words(content_in_dir)
            store_common(counts, args.n_words, "n_words.csv")
