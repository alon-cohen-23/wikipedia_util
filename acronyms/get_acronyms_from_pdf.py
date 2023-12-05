import regex
from tqdm import tqdm
from string import punctuation, whitespace
import json

# in order to simplify the retrieval of info from the pdf, we manually copied it to txt file.
def read_lines_from_file(pdf_copied_file):
    filename = pdf_copied_file # '''acronyms\\acronyms_copied.txt'''
    with open(filename, encoding='utf-8') as f:
        lines = f.readlines()
    return lines

def remove_comments(txt):
    to_remove = [(')', '('), (']', '[')]  # , (']', ']')]
    for tr in to_remove:
        pattern = fr'\{tr[0]}[^{tr[1]}]*\{tr[1]}'
        txt = regex.sub(pattern, "", txt)
    return txt

def is_nikud(c):
    # notepad++ regular expression search: [\x{0591}-\x{05BD}\x{05BF}-\x{05C2}\x{05C4}-\x{05C7}]
    ord_c = ord(c)
    return 0x0591 <= ord_c <= 0x5BD or 0x5BF <= ord_c <= 0x5C2 or 0x5C4 <= ord_c <= 0x5C7


def remove_nikud(txt):
    new_txt = ''
    i = 0
    while True:
        if i == len(txt): break
        if is_nikud(txt[i]):
            i += 1
            # special handling for cases where right after the nikud there is a whitespace - caused by pdf copying
            if i < (len(txt) - 1) and ord(txt[i]) == 32:
                i += 1
            continue
        new_txt += txt[i]
        i += 1
    return new_txt

def clean_and_retrieve_acronyms(lines):
    lines = [remove_nikud(line) for line in lines]

    pattern_title1 = '\p{Hebrew}+\"\p{Hebrew}\p{Hebrew}*$'
    pattern_title2 = '\p{Hebrew}\'?\/?$'

    acronyms = []
    for i in tqdm(range(len(lines))):
        line = lines[i]
        if regex.match(pattern_title1, line) or regex.match(pattern_title2, line):
            title_index = i
            acronyms.append({'title': line.strip(), 'vals': []})
        else:
            if len(acronyms) == 0:
                print(f'ERROR! {line}')
            else:
                content_bullet_pattern = '\d+ ?\. '
                if regex.match(content_bullet_pattern, line):
                    line = regex.sub(content_bullet_pattern, "", line)
                    acronyms[-1]['vals'].append(line.strip())
                else:
                    if len(acronyms[-1]['vals']) == 0:
                        acronyms[-1]['vals'].append(line.strip())
                    else:
                        acronyms[-1]['vals'][-1] = acronyms[-1]['vals'][-1] + ' ' + line.strip(
                            ''.join(list(punctuation) + list(whitespace)))

    for i in range(len(acronyms)):
        acronyms[i]['vals'] = [remove_comments(x) for x in acronyms[i]['vals']]

    acronyms_dict = {d['title']: d['vals'] for d in acronyms}
    return acronyms_dict

if __name__=='__main__':
    filename = '''data\\acronyms_copied.txt'''
    output_file = '''data\\output_acronyms_from_pdf.json'''
    lines = read_lines_from_file(filename)
    acronyms_dict = clean_and_retrieve_acronyms(lines)
    for k in acronyms_dict:
        print(f'--- {k} ---')
        print(acronyms_dict[k])
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(acronyms_dict, f, ensure_ascii=False)