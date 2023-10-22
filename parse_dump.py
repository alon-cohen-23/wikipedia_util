# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 17:05:05 2023

@author: DEKELCO
"""

import re
import mwxml
import mwparserfromhell

RE_BOLD = re.compile(r"'''(.*?)'''",re.MULTILINE | re.IGNORECASE)

def process_dump(dump, path):
  for page in dump:
      yield page

def extract_rev_first_para(rev_text):
    """        
    rev_text : Wiki markup of a wikipedia page 

    Returns
    -------
    text of the first section
    """
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
  dump_path = r'/Users/aloncohen/Documents/wikipedia_util/Documentswikipedia_util'
  paths = [dump_path]
  dump_gen = mwxml.map(process_dump, paths)
  for page in dump_gen:
      if 'מרלן דיטריך' in page.title:
          print('***',page.title)
          first_para_lines = extract_page_first_para(page)
          print('***BOLD:',extract_first_first_bold_span_from_1st_sent(first_para_lines))          
          print('***', first_para_lines)          
          break
      

if __name__ == '__main__':
  main()