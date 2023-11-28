import time
import pandas as pd
from loguru import logger
import wikipediaapi

MAX_RETRIES = 3


def get_pages_and_subcategories(top_categories, max_depth=6, language='he'):
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36 Edg/92.0.902.78'

    wiki_wiki = wikipediaapi.Wikipedia(language=language, user_agent=user_agent)

    def collect_all_pages_and_descendants(category, page_list, visited_pages, visited_categories, depth):
        if depth == 0:
            return

        if category.title in visited_categories:
            return

        visited_categories.add(category.title)
        print(f"Subcategory: {category.title}")
        subcategories = category.categorymembers
        for subcategory in subcategories.values():
            num_retries = 0
            while True:
                try:
                    if subcategory.ns == wikipediaapi.Namespace.CATEGORY:
                        collect_all_pages_and_descendants(subcategory, page_list, visited_pages, visited_categories,
                                                          depth - 1)
                    elif subcategory.ns == wikipediaapi.Namespace.MAIN:
                        if subcategory.title not in visited_pages:
                            page_list.append(subcategory.title)
                            visited_pages.add(subcategory.title)
                            print(f"Page: {subcategory.title}")
                    break  # on successful attempt - break from while True retries loop (no retries)
                except Exception as e:
                    num_retries += 1
                    if num_retries == MAX_RETRIES:
                        logger.error(f"Failed to process {subcategory} after {MAX_RETRIES} retries. Skipping.")
                        break
                    logger.warning(f"Error processing {subcategory}, retrying ({num_retries} of {MAX_RETRIES}): {e}")
                    time.sleep(5)

    all_pages = []
    visited_pages = set()
    visited_categories = set()

    for category_name in top_categories:
        try:
            category = wiki_wiki.page(category_name)
        except:
            logger.info(f'Failed to find the category: {category_name}. Skipping it and all its sub-categories')
            continue

        collect_all_pages_and_descendants(category, all_pages, visited_pages, visited_categories, depth=max_depth)

    return all_pages, list(visited_categories)


# Example usage:
lang = 'ar'
top_categories = {'ar': ["تصنيف:عسكرية", "تصنيف:نزاع_عنيف", "تصنيف:نزاع", "تصنيف:الشرق_الأوسط"],
                  'he': ["קטגוריה:צבא", "קטגוריה:לחימה", "קטגוריה:סכסוכים", "קטגוריה:המזרח_התיכון"]
                  }
# top_categories =
pages, subcategories = get_pages_and_subcategories(top_categories[lang], language=lang)

# Create dataframes for pages and subcategories
pages_df = pd.DataFrame({"Page": pages})
subcategories_df = pd.DataFrame({"Subcategory": subcategories})

# Define paths for the Parquet files
pages_file_path = f"relevant_pages_{lang}.parquet"
subcategories_file_path = f"relevant_subcategories_{lang}.parquet"

# Write dataframes to Parquet files
pages_df.to_parquet(pages_file_path)
subcategories_df.to_parquet(subcategories_file_path)
