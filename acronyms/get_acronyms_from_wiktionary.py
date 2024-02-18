# importing the modules
import requests, os
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

url = 'https://he.wiktionary.org/wiki/%D7%A7%D7%98%D7%92%D7%95%D7%A8%D7%99%D7%94:%D7%A8%D7%90%D7%A9%D7%99_%D7%AA%D7%99%D7%91%D7%95%D7%AA_%D7%91%D7%A6%D7%94%22%D7%9C'


def get_acronym_meaning_from_wiki_page(name, wiki_page):
    def clean_meaning(meaning):
        return meaning.split('.')[0].replace(idf_string, '').split('\n')[0].split('(')[0].strip()

    idf_string = '(צה"ל)'

    req = requests.get(wiki_page)
    soup = BeautifulSoup(req.text, 'html')

    section = soup.find(class_='mw-content-rtl mw-parser-output')
    oul = section.find(['ol', 'ul'])
    if oul is not None:
        # if page contains bullets:
        lis = oul.find_all('li')
        if lis is not None and len(lis) > 0:
            # check if one of them contain 'צה"ל' in 'small' font
            for li in lis:
                small = li.find('small')
                if small is not None and small.text == idf_string:
                    selected_li = li
                    break
            else:
                # take the first one
                selected_li = lis[0]
            meaning = selected_li.text
        else:
            print(f'**** {name} {wiki_page} ****')
            meaning = None
    else:
        p = section.find('p')
        if p:
            meaning = p.text
        else:
            print(f'%%%% {name} {wiki_page} ****')
            meaning = None
    if meaning:
        meaning = clean_meaning(meaning)
    return meaning


# get_acronym_meaning_from_wiki_page('', 'https://he.wiktionary.org/wiki/%D7%90%D7%92%D7%9E%22%D7%A8')
def retrieve_acronyms_meaning_from_wiki(wiki_url, start_url):
    # acronyms_list = []
    d_acronyms = {}
    url = start_url
    while True:
        print(url)
        req = requests.get(url)
        soup = BeautifulSoup(req.text, 'html')

        section = soup.find(class_='mw-category mw-category-columns')
        uls = section.find_all('ul')
        for ul in tqdm(uls): # first letters
            lis = ul.find_all('li')
            for li in lis: # acronyms
                href = wiki_url + li.a.get('href')
                name = li.text
                meaning = get_acronym_meaning_from_wiki_page(name, href)
                # print(name, ' ---- ', meaning)
                d_acronyms[name] = meaning

        try:
            next_page = soup.find('a', href=True, text='לדף הבא')
            url = wiki_url + next_page['href']
        except:
            break
    return d_acronyms




if __name__=='__main__':
    base_url = 'https://he.wiktionary.org/'
    start_url = 'https://he.wiktionary.org/wiki/%D7%A7%D7%98%D7%92%D7%95%D7%A8%D7%99%D7%94:%D7%A8%D7%90%D7%A9%D7%99_%D7%AA%D7%99%D7%91%D7%95%D7%AA_%D7%91%D7%A6%D7%94%22%D7%9C'
    d_acronyms = retrieve_acronyms_meaning_from_wiki(base_url, start_url)
    df = pd.DataFrame.from_dict(d_acronyms, 'index', columns=['meaning'])
    output_file = 'data\\outputs\\output_acronyms_from_wiktionary.csv'
    df.to_csv(output_file)