import pandas as pd
import wikipediaapi

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
            if subcategory.ns == wikipediaapi.Namespace.CATEGORY:
                collect_all_pages_and_descendants(subcategory, page_list, visited_pages, visited_categories, depth - 1)
            elif subcategory.ns == wikipediaapi.Namespace.MAIN:
                if subcategory.title not in visited_pages:
                    page_list.append(subcategory.title)
                    visited_pages.add(subcategory.title)
                    print(f"Page: {subcategory.title}")

    all_pages = []
    visited_pages = set()
    visited_categories = set()

    for category_name in top_categories:
        category = wiki_wiki.page(category_name)
        collect_all_pages_and_descendants(category, all_pages, visited_pages, visited_categories, depth=max_depth)

    return all_pages, list(visited_categories)

# Example usage:
top_categories = ["קטגוריה:צבא","קטגוריה:לחימה", "קטגוריה:סכסוכים", "קטגוריה:המזרח_התיכון"]
pages, subcategories = get_pages_and_subcategories(top_categories)

# Create dataframes for pages and subcategories
pages_df = pd.DataFrame({"Page": pages})
subcategories_df = pd.DataFrame({"Subcategory": subcategories})

# Define paths for the Parquet files
pages_file_path = "pages.parquet"
subcategories_file_path = "subcategories.parquet"

# Write dataframes to Parquet files
pages_df.to_parquet(pages_file_path)
subcategories_df.to_parquet(subcategories_file_path)

