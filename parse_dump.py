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
from df_process import filter_sentences_df

RE_BOLD = re.compile(r"'''(.*?)'''", re.MULTILINE | re.IGNORECASE)


def process_dump(dump, path):
    for page in dump:
        yield page


dump_path = r'path/to/xml'
paths = [dump_path]
dump_gen = mwxml.map(process_dump, paths)


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


def main(pages_df_path):
    """
    Returns
    -------
    sentences_df : df that contains 2 columns: HE_sentences: contains the text of the relevant values based on the given df_path devided to sentences.
    title: the title of the wikipedia value that contain the given sentence.

    """
    pages_df = pd.read_parquet(pages_df_path)

    pages = pages_df['Page'].to_list()
    data_frames = []  # Create a list to store DataFrames

    relevant_values_df = pd.read_parquet('categories_pages/he/pages.parquet')
    relevant_values_titles = relevant_values_df['Page'].to_list()

    for index, page in enumerate(dump_gen):

        try:
            if page.title in relevant_values_titles:
                page_wikicode = extract_page_wikicode(page)
                page_sentences = extract_sentences_from_wikicode(page_wikicode)

                data_frames.append(pd.DataFrame({'title': page.title, 'HE_sentences': page_sentences}))

        except:
            print('failed to load {pag.title}')

        if index % 10000 == 0:
            print(f'already iterated over {index} values')

    sentences_df = pd.concat(data_frames, ignore_index=True)
    sentences_df = filter_sentences_df(df)  # filter the df by calling the function from df_process.py

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
    df = main()
    df.to_parquet('relevant_categories_sentences.parquet')

    """df = filter_sentences_df (df)
    split_df (df)
    """
    pages_df_path = 'categories_pages/he/pages.parquet'
    df = main(pages_df_path)
