import time
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
import os
from pathlib import Path
from slugify import slugify
import logging

options = uc.options.ChromeOptions()
options.headless = False
# Create a new Chrome driver instance
browser = uc.Chrome(options)

logger = logging.getLogger(__name__)

def prepare_list_of_pubs(main_url, output_dir, links_filename):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, links_filename)

    browser.get(main_url)
    time.sleep(5)

    # Collect all links: try to scroll down until no new links are added
    last_links = []
    count_no_progress = 0
    while True:
        # browser must be in focus, otherwise scrolling is stopped
        browser.execute_script("window.focus();")
        javaScript = "window.scrollBy(0, window.innerHeight);"
        browser.execute_script(javaScript)
        time.sleep(5)

        # links are located in the below elements
        elements = WebDriverWait(browser, 60).until(EC.visibility_of_all_elements_located((By.CLASS_NAME, "read_more_p")))
        links = [elem.get_attribute('href') for elem in elements]

        print(len(links), end="-")
        if (len(links)==len(last_links)) and len(links)>10:
            count_no_progress+=1
            if count_no_progress>10: # no progress for many iterations
                break
        else:
            count_no_progress=0

        if len(links)>100:
            with open(output_path, 'w') as f:
                f.write('\n'.join(links))
        last_links = links[:]

    print(f'\nFound {len(links)} links')
    with open(output_path, 'w') as f:
        f.write('\n'.join(links))

def get_text_from_publication(link:str, output_dir):
    # # https://www.inss.org.il/he/publication/gulf-states-hamas
    # options = uc.options.ChromeOptions()
    # options.headless = False
    # # Create a new Chrome driver instance
    # browser = uc.Chrome(options)
    # browser.get("http://www.ynet.co.il")
    browser.get(link)
    time.sleep(2)

    try:
        content_element = browser.find_element(By.CLASS_NAME,"content_publication_post")

    except NoSuchElementException:
        logger.warning(f'{slugify(link)} NoSuchElementException')
        return 0

    txt = content_element.text
    if len(txt)>0:
        filename = slugify(link.split('/')[-2])[:30]
        with open(os.path.join(output_dir, f'{filename}.txt'), 'w', encoding="utf-8") as f:
            f.write(txt)

    return len(txt)>0


def get_texts_from_links(links_filename, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(links_filename, 'r') as f:
        links = [x.strip() for x in f.readlines()]

    l_txts = []
    l_others = []
    n_txts = 0
    for l in links:
        print(l)
        is_txt = get_text_from_publication(l,output_dir)
        if is_txt:
            l_txts.append(l)
        else:
            l_others.append(l)
        logger.info(f'{slugify(l)} is_text: {is_txt}')
        n_txts+= is_txt
    print(f'{len(links)} links were processed. {n_txts} text were found')
    with open(os.path.join(output_dir, 'txt_files.txt'), 'w') as f:
        f.write('\n'.join(l_txts))
    with open(os.path.join(output_dir, 'other_files.txt'), 'w') as f:
        f.write('\n'.join(l_others))

if __name__=='__main__':
    inss_url_prefix = "https://www.inss.org.il/he/publication/?ptype="
    output_dir = '.\\retrieved_data'
    links_filename = 'links.txt'
    prepare_list_of_pubs(inss_url_prefix, output_dir, links_filename)
    get_texts_from_links(os.path.join(output_dir, links_filename), os.path.join(output_dir, 'texts'))

    # get_text_from_publication('https://www.inss.org.il/he/publication/gulf-states-hamas')
    # get_text_from_publication('https://www.inss.org.il/he/publication/%d7%a7%d7%95-%d7%94%d7%92%d7%a0%d7%94-%d7%91%d7%99%d7%94%d7%95%d7%93%d7%94-%d7%95%d7%91%d7%a9%d7%95%d7%9e%d7%a8%d7%95%d7%9f/')
