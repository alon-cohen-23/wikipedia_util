# importing the modules
import requests, os
from bs4 import BeautifulSoup
from wiktionaryparser import WiktionaryParser

parser = WiktionaryParser()

url = 'https://he.wiktionary.org/wiki/%D7%A7%D7%98%D7%92%D7%95%D7%A8%D7%99%D7%94:%D7%A8%D7%90%D7%A9%D7%99_%D7%AA%D7%99%D7%91%D7%95%D7%AA_%D7%91%D7%A6%D7%94%22%D7%9C'

# for now, we just retrieve the acronyms (without their meaning)
# TODO: add acronyms meaning retrieval

def retrieve_acronyms_list_from_wiktionary(base_url, start_url):
    acronyms_list = []
    url = start_url
    while True:
        print(url)
        req = requests.get(url)
        soup = BeautifulSoup(req.text, 'html')

        section = soup.find(class_='mw-category mw-category-columns')
        uls = section.find_all('ul')
        for ul in uls:
            lis = ul.find_all('li')
            for li in lis:
                acronyms_list.append(li.text)
                print(li.text)

        print('----')
        try:
            next_page = soup.find('a', href=True, text='לדף הבא')
            url = base_url + next_page['href']
        except:
            break
    return acronyms_list

if __name__=='__main__':
    base_url = 'https://he.wiktionary.org/'
    start_url = 'https://he.wiktionary.org/wiki/%D7%A7%D7%98%D7%92%D7%95%D7%A8%D7%99%D7%94:%D7%A8%D7%90%D7%A9%D7%99_%D7%AA%D7%99%D7%91%D7%95%D7%AA_%D7%91%D7%A6%D7%94%22%D7%9C'
    acronyms_list = retrieve_acronyms_list_from_wiktionary(base_url, start_url)
    output_file = 'data\\output_acronyms_from_wiktionary.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(acronyms_list))