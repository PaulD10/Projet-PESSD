from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import pandas as pd
from tqdm import tqdm

# List of categories and their corresponding URLs
categories_urls = [
    ("Economie générale", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=28&categorie=2&collection=116"),
    ("Conjoncture", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=30&categorie=2&collection=116"),
    ("Comptes nationaux trimestriels", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=32&categorie=2&collection=116"),
    ("Comptes nationaux annuels", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=33&categorie=2&collection=116"),
    ("Finances publiques", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=36&categorie=2&collection=116"),
    ("Commerce extérieur", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=35&categorie=2&collection=116"),
    ("Evolution et structure de la population", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=1&categorie=2&collection=116"),
    ("Naissances - Fécondité", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=2&categorie=2&collection=116"),
    ("Décès - Mortalité - Espérance de vie", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=3&categorie=2&collection=116"),
    ("Couple - Familles - Ménages", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=4&categorie=2&collection=116"),
    ("Etrangers - Immigrés", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=5&categorie=2&collection=116"),
    ("Revenus - Niveau de vie - Pouvoir d'achat - Consommation", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=81&categorie=2&collection=116"),
    ("Protection sociale - Retraites", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=26&categorie=2&collection=116"),
    ("Pauvreté - Précarité", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=82&categorie=2&collection=116"),
    ("Patrimoine", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=83&categorie=2&collection=116"),
    ("Consommation et équipement des ménages", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=19&categorie=2&collection=116"),
    ("Société - Vie sociale - Elections", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=10&categorie=2&collection=116"),
    ("Education - Formation - Compétences", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=15&categorie=2&collection=116"),
    ("Logement", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=7&categorie=2&collection=116"),
    ("Egalité femmes-hommes", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=84&categorie=2&collection=116"),
    ("Santé - Handicap - Dépendance", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=11&categorie=2&collection=116"),
    ("Sécurité - Justice", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=8&categorie=2&collection=116"),
    ("Loisirs - Culture", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=9&categorie=2&collection=116"),
    ("Emploi - Population active", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=22&categorie=2&collection=116"),
    ("Chômage", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=23&categorie=2&collection=116"),
    ("Salaires et revenus d'activité", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=24&categorie=2&collection=116"),
    ("Entreprises", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=38&categorie=2&collection=116"),
    ("Démographie et création des entreprises", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=38&categorie=2&collection=116"),
    ("Caractéristiques des entreprises", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=39&categorie=2&collection=116"),
    ("Mondialisation, compétitivité et innovation", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=43&categorie=2&collection=116"),
    ("Agriculture", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=45&categorie=2&collection=116"),
    ("Commerce", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=49+51+52+48&categorie=2&collection=116"),
    ("Industrie", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=54+57+56+55+58+53&categorie=2&collection=116"),
    ("Construction", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=59&categorie=2&collection=116"),
    ("Services", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=61+63+60&categorie=2&collection=116"),
    ("Transports", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=66&categorie=2&collection=116"),
    ("Tourisme", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=67&categorie=2&collection=116"),
    ("Economie sociale et solidaire", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=85&categorie=2&collection=116"),
    ("Equipements et services à la population", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=69&categorie=2&collection=116"),
    ("Villes et quartiers", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=71&categorie=2&collection=116"),
    ("Dynamique des territoires", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=72&categorie=2&collection=116"),
    ("Mobilités - Déplacements - Frontaliers", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=73&categorie=2&collection=116"),
    ("Environnement", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=75&categorie=2&collection=116"),
    ("Développement durable", "https://www.insee.fr/fr/statistiques?taille=100&debut=0&theme=76&categorie=2&collection=116")
]

def scrape_links(category, url):
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run in headless mode
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    # Initialize the driver
    driver = webdriver.Chrome(options=chrome_options)

    data = []
    try:
        # Navigate to the page
        driver.get(url)

        # Find all elements with class 'echo-lien', 'echo-titre', and 'echo-chapo'
        links = driver.find_elements(By.CLASS_NAME, 'echo-lien')
        titles = driver.find_elements(By.CLASS_NAME, 'echo-titre')
        abstracts = driver.find_elements(By.CLASS_NAME, 'echo-chapo')

        # Extract data and filter out rows with titles containing "anciens numéros"
        for link, title, abstract in zip(links, titles, abstracts):
            href = link.get_attribute('href')
            title_text = title.text
            abstract_text = abstract.text

            if href and 'anciens numéros' not in title_text.lower():
                full_url = href
                data.append((category, full_url, title_text, abstract_text))
    finally:
        driver.quit()

    return data

# List to store the data
data = []

# Iterate over each category and URL with a progress bar
for category, url in tqdm(categories_urls, desc="Scraping categories"):
    data.extend(scrape_links(category, url))


# Create a DataFrame
df = pd.DataFrame(data, columns=['Category', 'Link', 'Title', 'Abstract'])
df = df.sort_values('Category')
# Display the DataFrame
print(df)

# Save the DataFrame to a CSV file
df.to_csv('insee_links.csv', index=False)
