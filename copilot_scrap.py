import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import logging
import re

OUTPUT_DIR = "insee_publications_theme"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_text(text):
    # Replace special characters with a space
    text = re.sub(r'[^\w\s\'-]', ' ', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Replace multiple spaces/newlines with a single space
    text = re.sub(r'\s+', ' ', text)
    # Convert to lowercase
    text = text.lower()
    return text.strip()

def get_content_from_onglets(soup):
    # Try all divs with id starting with "onglet-" except "Documentation"
    onglet_divs = soup.find_all('div', id=re.compile(r'^onglet-\d+$'))
    for div in onglet_divs:
        if "Documentation" not in div.get('class', []):
            content_element = div.find(class_='corps-publication')
            if content_element:
                content = content_element.get_text(separator=' ', strip=True)
                if content:
                    logging.debug(f"Found content in div id: {div['id']}")
                    return content
    return ""

def download_publication(url):
    response = requests.get(url)
    if response.status_code != 200:
        logging.error(f"Failed to download publication: {url} - {response.status_code}")
        return None
    soup = BeautifulSoup(response.content, 'html.parser')
    content_element = soup.find(class_='corps-publication')
    content = content_element.get_text(separator=' ', strip=True) if content_element else ""

    if not content:  # if content is empty, try other onglets except "Documentation"
        logging.debug("Content is empty, trying other onglets.")
        content = get_content_from_onglets(soup)

    cleaned_content = clean_text(content)
    return cleaned_content

def process_csv(file_path, publication_type):
    df = pd.read_csv(file_path, encoding='utf-8')
    full_texts = []
    years = []
    theme = []

    for index, row in df.iterrows():
        url = row['url']
        year = row['date'].split('-')[0]  # Extract the year from the date
        logging.info(f"Scraping {publication_type} publication from {url}")
        publication_content = download_publication(url)
        if publication_content:
            full_texts.append(publication_content)
            years.append(year)
        else:
            full_texts.append("")
            years.append(year)

    output_df = pd.DataFrame({
        'year': years,
        'full_text': full_texts,
        'theme': theme
    })

    # Remove rows with empty full_text
    output_df = output_df[output_df['full_text'] != ""]

    output_file_path = os.path.join(OUTPUT_DIR, f"{publication_type}_full_texts.csv")
    output_df.to_csv(output_file_path, index=False, encoding='utf-8')
    logging.info(f"Saved full texts to {output_file_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    process_csv('insee_analyse_publications_full.csv', 'insee_analyse')
    process_csv('insee__premiere_publications_full.csv', 'insee_premiere')

if __name__ == "__main__":
    main()
