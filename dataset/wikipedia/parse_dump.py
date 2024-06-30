# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 17:05:05 2023

@author: DEKELCO
"""

import re
from pathlib import Path
import pandas as pd
import mwxml
import mwparserfromhell
from nltk.tokenize import sent_tokenize
from eng_utils.common.common_util import read_df_folder
from dataset.df_process import filter_sentences_df

RE_BOLD = re.compile(r"'''(.*?)'''", re.MULTILINE | re.IGNORECASE)


def process_dump(dump, path):
    for page in dump:
        yield page




def extract_rev_first_para(rev_text):
    wikicode = mwparserfromhell.parse(rev_text)
    # templates = wikicode.filter_templates()
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
    if page.redirect:  # Do not process redirect pages for now
        return first_para_lines

    for revision in page:  # hopefull only a single revision, as we exported only latest version
        first_para_lines = extract_rev_first_para(revision.text)
        break
    return first_para_lines


def extract_first_first_bold_span_from_1st_sent(lines):
    bold_spans = []
    if len(lines) > 0:
        matches = RE_BOLD.finditer(lines[0])
        for matchNum, match in enumerate(matches, start=1):
            # print ("Match {matchNum} was found at {start}-{end}: {match}".format(matchNum = matchNum, start = match.start(), end = match.end(), match = match.group()))

            for groupNum in range(0, len(match.groups())):
                groupNum = groupNum + 1
                print("Group {groupNum} found at {start}-{end}: {group}".format(groupNum=groupNum,
                                                                                start=match.start(groupNum),
                                                                                end=match.end(groupNum),
                                                                                group=match.group(groupNum)))
                bold_spans.append(match.group(groupNum))
    return bold_spans


def main(lang, dump_gen, pages_df_path):
    """
    Returns
    -------
    sentences_df : df that contains 2 columns: HE_sentences: contains the text of the relevant values based on the given df_path devided to sentences.
    title: the title of the wikipedia value that contain the given sentence.

    """
    data_frames = []  # Create a list to store DataFrames

    relevant_values_df = pd.read_parquet(pages_df_path)
    relevant_values_titles = relevant_values_df['Page'].to_list()

    for index, page in enumerate(dump_gen):

        try:
            if page.title in relevant_values_titles:
                page_wikicode = extract_page_wikicode(page)
                page_sentences = extract_sentences_from_wikicode(page_wikicode)
                #print('page_sentences',len(page_sentences))
                data_frames.append(pd.DataFrame({'title': page.title, 'HE_sentences': page_sentences}))

        except:
            print(f'failed to load {page.title}')

        if index > 0 and index % 10000 == 0:
            print(f'Extracted sentences from {index} pages')
        if index > 0 and index % 100000 == 0:
           print(f'Saving temp Extracted sentences from {index} pages') 
           sentences_df = pd.concat(data_frames, ignore_index=True)     
           sentences_df.to_parquet('./relevant_sentences_temp.parquet')    
    sentences_df = pd.concat(data_frames, ignore_index=True)    
    sentences_df = filter_sentences_df(sentences_df, lang)  # filter the df by calling the function from df_process.py

    return sentences_df


def extract_sentences_from_wikicode(wiki_text):
    """

    Parameters
    ----------
    wiki_text : wikicode text.

    Returns
    -------
    sentences : parse the wikicode to regular text and slit it to sentences.

    """
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
    if page.redirect:  # Do not process redirect pages for now
        return page_wikicode

    for rev in page:  # hopefull only a single revision, as we exported only latest version
        page_wikicode = mwparserfromhell.parse(rev.text)
        break
    return page_wikicode


if __name__ == '__main__':
    """
    Input: wikipedia dump + pages.parquet - list of relevant pages --> Output: Sentences from the relevant pages
    """ 
    lang = 'he' # 'fa' # Set language code here. Download corresponding dump from 
    dump_path = fr'wikipedia_dumps/{lang}wiki-latest-pages-articles.xml.bz2'
    # If added more data --> filter out sentences already translated, to save google translate time 
      # Set to None to disable filtering
    FILTER_ALREADY_TRANSLATED_FOLDER = f'translated/{lang}'
    paths = [dump_path]
    dump_gen = mwxml.map(process_dump, paths)
    pages_df_path=f'categories_pages/{lang}/pages.parquet' # Input: the pages df of all pages under categories (recursive - see wikipedia-api)
    df = main(lang, dump_gen, pages_df_path)
    p_out=Path(f'relevant_categories_sentences/{lang}')
    p_out.mkdir(parents=True, exist_ok=True)
    if not FILTER_ALREADY_TRANSLATED_FOLDER is None:
        df_translated = read_df_folder(FILTER_ALREADY_TRANSLATED_FOLDER) 
        df = df[~df['HE_sentences'].isin(df_translated['HE_sentences'])]
    df.to_parquet(p_out / f'relevant_categories_sentences_{lang}.parquet')

    