import numpy as np
import pandas as pd
import gensim
import spacy
import re
from gensim.models import Word2Vec
from tqdm.auto import tqdm

class INSEETextEmbedding:
    def __init__(self, min_word_freq=2, max_words=5000000):
        print("Initializing text analyzer...")
        self.nlp = spacy.load('fr_core_news_lg', disable=["ner", "parser"])
        self.min_word_freq = min_word_freq
        self.max_words = max_words
        self.stop_words = set(self.nlp.Defaults.stop_words)
        self.stop_words.update(["insee", "figure", "être", "avoir", "ainsi", "dont", "etc"])
        self.processed_texts = []

    def clean_text(self, text):
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\[^0-9a-zA-Z]+','',text)
        text = ' '.join(text.split()).lower()

        return text

    def process_texts(self, texts):
        processed_texts = []
        for text in tqdm(texts, desc="Processing texts"):
            doc = self.nlp(self.clean_text(text))
            tokens = []
            for token in doc:
                    if (token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN'] and
                        not token.is_stop and
                        not token.is_punct and
                        len(token.text) > 2 and
                        token.lemma_ not in self.stop_words):
                        tokens.append(token.lemma_)
            if tokens:
                processed_texts.append(tokens)
        print(processed_texts)
        return processed_texts

    def train_word_embeddings(self, processed_texts, categories, model_type='word2vec'):
        print(f"Training {model_type} model with category tracking...")
        model_class = Word2Vec if model_type == 'word2vec' else FastText
        model = model_class(sentences=processed_texts, vector_size=200, window=10, min_count=self.min_word_freq, sg=1, workers=4)

        # Create a dictionary to track word-category occurrences
        word_category_counts = {}

        # Count occurrences of each word in each category
        for text_tokens, category in zip(processed_texts, categories):
            for token in text_tokens:
                if token not in word_category_counts:
                    word_category_counts[token] = {}

                if category not in word_category_counts[token]:
                    word_category_counts[token][category] = 0

                word_category_counts[token][category] += 1

        # Find the most common category for each word
        word_most_common_category = {}
        for word, category_counts in word_category_counts.items():
            most_common_category = max(category_counts.items(), key=lambda x: x[1])[0]
            word_most_common_category[word] = most_common_category

        # Add the most common category as an attribute to the model
        model.wv.word_most_common_category = word_most_common_category

        return model

# Process texts
def main(df, period):
    texts = df["Full_Text"].dropna()
    categories = df["Category"]

    embedding_model = INSEETextEmbedding()
    processed_texts = embedding_model.process_texts(texts)
    word2vec_model = embedding_model.train_word_embeddings(processed_texts, categories, model_type='word2vec')
    word2vec_model.save(f"word2vec_insee_premiere_complet{period}.model")

df = pd.read_csv("/home/paul/Desktop/INSEE projet stat/output/insee_publications_complete/insee_publications_premiere_complete.csv")

df_2010_2014 = df[(df['Year'] >= 2010) & (df['Year'] <= 2014)]
df_2015_2019 = df[(df['Year'] >= 2015) & (df['Year'] <= 2019)]
df_2020_2024 = df[(df['Year'] >= 2020) & (df['Year'] <= 2024)]

main(df, "2010-2024")
main(df_2010_2014, "2010-2014")
main(df_2015_2019, "2015-2019")
main(df_2020_2024, '2020-2024')

# ##TEST DE MON MODELE
# # Get word embeddings
# vector = word2vec_model.wv["agglomération"]
# print(vector)  # Prints word embedding for "croissance"
#
# # Find similar words
# similar_words = word2vec_model.wv.most_similar("économie", topn=5)
# print(similar_words)
#
# print(f"Most common category for 'croissance': {word2vec_model.wv.word_most_common_category.get('agglomération', 'Not found')}")
# print(f"Most common category for 'économie': {word2vec_model.wv.word_most_common_category.get('économie', 'Not found')}")
#
# def get_similar_words_by_category(model, word, category, topn=10):
#     similar_words = model.wv.most_similar(word, topn=100)  # Get more than needed to allow filtering
#     category_filtered = [(w, score) for w, score in similar_words
#                          if model.wv.word_most_common_category.get(w) == category]
#     return category_filtered[:topn]  # Return only the top n after filtering
#
# # Example usage:
# economy_related = get_similar_words_by_category(word2vec_model, "économie", "Économie", topn=5)
# print(f"Similar words to 'économie' in 'Économie' category: {economy_related}")
