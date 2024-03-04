import glob
from pathlib import Path
import os
import regex

def get_subtitles_from_dir(base_dir, output_dir, program, n_seasons):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for i in range(1,n_seasons+1):
        folder = os.path.join(base_dir, f'{program}_S0{i}_heb')
        print(folder)
        for input_file in glob.glob(os.path.join(os.path.join(folder,'*.srt'))):
            pattern_se = "S(?P<season>\d{2})E(?P<episode>\d{2})"
            m = regex.search(pattern_se, input_file)
            output_file = os.path.join(output_dir, f"{program}_S{m['season']}E{m['episode']}.txt")
            get_subtitles(input_file, output_file)


def get_subtitles(filename, output_file, lang='heb'):
    if lang == 'heb':
        with open(filename, encoding="ISO-8859-8") as f:
            lines = f.readlines()
    else:
        with open(filename) as f:
            lines = f.readlines()

    clean_txt = []
    for line in lines:
        if regex.search('^[0-9]+$', line) is None and regex.search('^[0-9]{2}:[0-9]{2}:[0-9]{2}',
                                                                   line) is None and regex.search('^$', line) is None:
            clean_txt.append(line.strip())

    # move the punctation from start to end
    for i in range(len(clean_txt)):
        if clean_txt[i][0] in '!?.,:':
            if len(clean_txt[i]) > 3 and clean_txt[i][:3] == '...':
                punc_last_index = 2
            else:
                punc_last_index = 0
            clean_txt[i] = clean_txt[i][punc_last_index + 1:] + clean_txt[i][:punc_last_index + 1]

    joined_txt = ' '.join(clean_txt)
    pattern = "[^!.?]+[!.?]"
    sentences = regex.findall(pattern, joined_txt)
    with open(output_file, 'w') as f:
        f.writelines(sentences)
        # for s in sentences:
        #     f.write(s)

if __name__=='__main__':
    base_dir = '''C:\\Users\MICHALD2\\OneDrive - Rafael\\series_data'''
    output_dir = 'C:\\Users\MICHALD2\\OneDrive - Rafael\\series_data\\Fauda_sentences'
    get_subtitles_from_dir(base_dir, output_dir, 'Fauda', 4)