from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
from datetime import datetime

def get_insee_articles(url="https://www.insee.fr/fr/statistiques?debut=0&idprec=8283129&collection=6"):
    """
    Get URLs of articles from INSEE using Selenium

    Args:
        url (str): The URL of the INSEE statistics page with collection filter

    Returns:
        list: List of article URLs
    """
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run in headless mode
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    # Initialize the driver
    driver = webdriver.Chrome(options=chrome_options)

    try:
        # Navigate to the page
        driver.get(url)

        # Wait for the documents table to be present
        table = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "documents"))
        )

        # Wait a bit for dynamic content to load
        time.sleep(2)

        # Find all article links
        links = table.find_elements(By.TAG_NAME, "a")

        # Extract URLs
        urls = []
        for link in links:
            href = link.get_attribute('href')
            if href and 'statistiques' in href:
                urls.append(href)

        return list(dict.fromkeys(urls))  # Remove duplicates while preserving order

    finally:
        # Clean up
        driver.quit()

def get_all_collection_pages():
    """
    Get all article URLs across multiple pages
    """
    all_urls = []
    page = 0

    while True:
        url = f"https://www.insee.fr/fr/statistiques?taille=100&debut={page * 100}&collection=6"
        print(f"Fetching page {page + 1}...")

        urls = get_insee_articles(url)

        if not urls or urls[-1] in all_urls:  # No new URLs found
            break

        all_urls.extend(urls)
        page += 1

    return list(dict.fromkeys(all_urls))  # Remove any duplicates

def save_urls_as_python_list(urls, filename=None):
    """
    Save URLs as a Python list of strings in a .py file

    Args:
        urls (list): List of URLs to save
        filename (str, optional): Name of the file to save to. If None, generates a name
    """
    if filename is None:
        # Generate filename with current date
        current_date = datetime.now().strftime("%Y%m%d")
        filename = f"insee_analyse_urls_{current_date}.py"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("urls = [\n")
        for url in urls:
            # Properly format each URL as a string with proper escaping
            f.write(f"    \"{url}\",\n")
        f.write("]\n")

    return filename

if __name__ == "__main__":
    print("Fetching Insee Analyse articles...")
    urls = get_all_collection_pages()

    # Save URLs to Python file
    filename = save_urls_as_python_list(urls)

    print(f"\nFound {len(urls)} articles.")
    print(f"URLs have been saved to: {filename}")
