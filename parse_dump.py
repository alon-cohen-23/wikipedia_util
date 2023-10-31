# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 17:05:05 2023

@author: DEKELCO
"""

import re
import mwxml
import mwparserfromhell
from nltk.tokenize import sent_tokenize
import pandas as pd
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


RE_BOLD = re.compile(r"'''(.*?)'''",re.MULTILINE | re.IGNORECASE)


def process_dump(dump, path):
  for page in dump:
      yield page

dump_path = r'H:\Alon\large_files/hewiki-latest-pages-articles.xml.bz2'
paths = [dump_path]
dump_gen = mwxml.map(process_dump, paths)

def extract_rev_first_para(rev_text):
    wikicode = mwparserfromhell.parse(rev_text)
    #templates = wikicode.filter_templates()
    # First paragraph bolds
    # First section, filtered from templates lines (infobox, about, cite ...)
    sections = wikicode.get_sections()
    sec_lines = sections[0].split('\n')
    sec_tpls = sections[0].filter_templates(recursive=False)
    tpl_lines = set('\n'.join([str(tpl) for tpl in sec_tpls]).split('\n'))
    text_lines = [line for line in sec_lines if not line in tpl_lines]
    return text_lines
    
def extract_page_first_para(page):    
    first_para_lines = None
    if page.redirect: # Do not process redirect pages for now
        return first_para_lines
    
    for revision in page: # hopefull only a single revision, as we exported only latest version                
        first_para_lines = extract_rev_first_para(revision.text)
        break
    return first_para_lines

def extract_first_first_bold_span_from_1st_sent(lines):    
    bold_spans = []
    if len(lines) > 0:
        matches = RE_BOLD.finditer(lines[0])
        for matchNum, match in enumerate(matches, start=1):    
            #print ("Match {matchNum} was found at {start}-{end}: {match}".format(matchNum = matchNum, start = match.start(), end = match.end(), match = match.group()))
            
            for groupNum in range(0, len(match.groups())):
                groupNum = groupNum + 1            
                print ("Group {groupNum} found at {start}-{end}: {group}".format(groupNum = groupNum, start = match.start(groupNum), end = match.end(groupNum), group = match.group(groupNum)))
                bold_spans.append(match.group(groupNum))
    return bold_spans
        
def clean_xml_string(xml_string):
    # Define a regular expression to match invalid characters and replace them with an empty string
    invalid_char_pattern = re.compile(r'[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD\u10000-\u10FFFF]+', re.UNICODE)
    return invalid_char_pattern.sub('', xml_string)
        
def main():
    data_frames = []  # Create a list to store DataFrames
        
    for index, page in enumerate(dump_gen):
        print('***',page.title)
        page_wikicode = extract_page_wikicode(page)
        page_sentences = extract_sentences_from_wikicode (page_wikicode)
        """
        print (len(page_sentences))
        count =0
        for item in page_sentences:
            if detect(item) == 'he':
                count = count +1
        print (count)
        break    
            """
        data_frames.append(pd.DataFrame({'HE_sentences': page_sentences}))
                
        if index >1000:
            break
    sentences_df = pd.concat(data_frames, ignore_index=True)
    return sentences_df    

def extract_sentences_from_wikicode (wiki_text):
     wikicode = mwparserfromhell.parse(wiki_text)
     text = str(wikicode.strip_code())
     
     lines = text.split('\n')
     sentences = []
     for line in lines:
         sentences.extend(sent_tokenize(line))
      
     sentences = [sentence for sentence in sentences if '|' not in sentence]
     return sentences 

def extract_page_wikicode(page):    
    page_wikicode = None
    if page.redirect: # Do not process redirect pages for now
        return page_wikicode
    
    for rev in page: # hopefull only a single revision, as we exported only latest version                
        page_wikicode = mwparserfromhell.parse(rev.text)
        break
    return page_wikicode          

def filter_sentences_df (df):
   # filter the length of the sentences (between 4-30 words)
   df['word_count'] = df['HE_sentences'].apply(lambda x: count_words(x))
   len_filtered_df = df.query('word_count > 3 and word_count < 31').copy()
   
   #filter the df to only hebrew sentences
   len_filtered_df['sen_lang'] = len_filtered_df['HE_sentences'].apply(lambda x: detect_lang(x))
   He_filtered_df = len_filtered_df.query("sen_lang == 'he'")
   
   return He_filtered_df
    
def count_words(cell_content):
    words = cell_content.split()
    return len(words)    

def detect_lang (cell_content):
    try:
        return detect(cell_content)
    except LangDetectException:
        return 'unknown'

def split_df (df):
    copied_df = df.copy()
   
    for number in range (int(len(df)/30000+1)): # the max size for google translate 
        if len(df)>30000:
            part_df = copied_df.head(30000)
            df = copied_df.drop(df.index[:30000])
        else:
            part_df = copied_df
            
        part_df.to_excel(f'he_tr_excel/{number}_part_HE.xlsx')    
            
if __name__ == '__main__':
    df = main()
    
    df = filter_sentences_df (df)
    print (df.columns)
    df = df.drop(columns=['word_count', 'sen_lang'])
    df.to_excel('1000_wikipedia_values_he.xlsx')
    split_df (df)
   
    #filtered_df.to_html('filter_output.html')
    """
    filtered_out_df = df[~df['HE_sentences'].isin(filtered_df['HE_sentences'])]
    filtered_out_df.to_html('filtered_out_df.html')
    """
    
    print (len(df)/30000)
